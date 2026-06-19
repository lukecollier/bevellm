use crate::{LlmResponseMode, LlmWorldContext};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{
        llama::{Cache, Config as LlamaRuntimeConfig, Llama, LlamaConfig, LlamaEosToks},
        quantized_llama::ModelWeights as QuantizedLlama,
        quantized_qwen2::ModelWeights as QuantizedQwen2,
    },
    utils::apply_repeat_penalty,
};
use log::{debug, info};
use rig_core::completion::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
use tokenizers::Tokenizer;

/// Supported local model configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmModel {
    /// HuggingFaceTB/SmolLM2-360M-Instruct.
    SmolLM2_360MInstruct,
    /// HuggingFaceTB/SmolLM2-1.7B-Instruct quantized to Q4_K_M GGUF.
    SmolLM2_1_7BInstructQ4KM,
    /// Qwen/Qwen2.5-1.5B-Instruct quantized to Q2_K GGUF.
    Qwen2_5_1_5BInstructQ2K,
    /// Qwen/Qwen2.5-1.5B-Instruct quantized to Q4_K_M GGUF.
    Qwen2_5_1_5BInstructQ4KM,
    /// HuggingFaceTB/SmolLM3-3B quantized to Q4_K_M GGUF.
    ///
    /// Candle 0.8.4 does not ship a SmolLM3 runtime backend yet, so selecting this
    /// variant currently returns an unsupported-model error at load time.
    SmolLM3_3BQ4KM,
}

impl LlmModel {
    fn repo_id(self) -> &'static str {
        match self {
            Self::SmolLM2_360MInstruct => "HuggingFaceTB/SmolLM2-360M-Instruct",
            Self::SmolLM2_1_7BInstructQ4KM => "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
            Self::Qwen2_5_1_5BInstructQ2K => "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            Self::Qwen2_5_1_5BInstructQ4KM => "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            Self::SmolLM3_3BQ4KM => "ggml-org/SmolLM3-3B-GGUF",
        }
    }

    fn revision(self) -> &'static str {
        "main"
    }

    fn tokenizer_repo_id(self) -> &'static str {
        match self {
            Self::SmolLM2_360MInstruct => self.repo_id(),
            Self::SmolLM2_1_7BInstructQ4KM => "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            Self::Qwen2_5_1_5BInstructQ2K => "Qwen/Qwen2.5-1.5B-Instruct",
            Self::Qwen2_5_1_5BInstructQ4KM => "Qwen/Qwen2.5-1.5B-Instruct",
            Self::SmolLM3_3BQ4KM => "HuggingFaceTB/SmolLM3-3B",
        }
    }

    fn source_kind(self) -> ModelSourceKind {
        match self {
            Self::SmolLM2_360MInstruct => ModelSourceKind::DenseSafetensors,
            Self::SmolLM2_1_7BInstructQ4KM => ModelSourceKind::QuantizedLlamaGguf {
                gguf_filename: "smollm2-1.7b-instruct-q4_k_m.gguf",
            },
            Self::Qwen2_5_1_5BInstructQ2K => ModelSourceKind::QuantizedLlamaGguf {
                gguf_filename: "qwen2.5-1.5b-instruct-q2_k.gguf",
            },
            Self::Qwen2_5_1_5BInstructQ4KM => ModelSourceKind::QuantizedLlamaGguf {
                gguf_filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            },
            Self::SmolLM3_3BQ4KM => ModelSourceKind::Unsupported {
                reason: "SmolLM3 uses a dedicated `smollm3` architecture, and Candle 0.8.4 does not provide a compatible runtime backend yet.",
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ModelSourceKind {
    DenseSafetensors,
    QuantizedLlamaGguf { gguf_filename: &'static str },
    Unsupported { reason: &'static str },
}

/// Runtime options for a named local Candle-backed LLM profile.
#[derive(Debug, Clone)]
pub struct LlmRuntimeProfileConfig {
    pub id: String,
    pub model: LlmModel,
    pub cache_dir: Option<PathBuf>,
    pub use_gpu: bool,
    pub worker_count: usize,
    pub seed: u64,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub max_new_tokens: usize,
}

impl Default for LlmRuntimeProfileConfig {
    fn default() -> Self {
        Self {
            id: String::from("default"),
            model: LlmModel::SmolLM2_360MInstruct,
            cache_dir: None,
            use_gpu: false,
            worker_count: 2,
            seed: 42,
            temperature: Some(0.7),
            top_p: Some(0.9),
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            max_new_tokens: 128,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmTaskRoutingConfig {
    pub turn_generation_profile: String,
    pub fact_extraction_profile: String,
}

impl Default for LlmTaskRoutingConfig {
    fn default() -> Self {
        Self {
            turn_generation_profile: String::from("default"),
            fact_extraction_profile: String::from("default"),
        }
    }
}

/// Runtime options for the local Candle-backed LLM worker profiles.
#[derive(Debug, Clone)]
pub struct LlmRuntimeConfig {
    pub profiles: Vec<LlmRuntimeProfileConfig>,
    pub routing: LlmTaskRoutingConfig,
}

impl Default for LlmRuntimeConfig {
    fn default() -> Self {
        Self {
            profiles: vec![LlmRuntimeProfileConfig::default()],
            routing: LlmTaskRoutingConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ValidatedLlmRuntimeConfig {
    pub profiles: Vec<LlmRuntimeProfileConfig>,
    pub routing: LlmTaskRoutingConfig,
}

impl LlmRuntimeConfig {
    pub fn validate(&self) -> Result<ValidatedLlmRuntimeConfig, LlmRuntimeError> {
        let mut seen = BTreeSet::new();
        let mut profile_ids = BTreeSet::new();
        let mut profiles = Vec::with_capacity(self.profiles.len());

        for profile in &self.profiles {
            let profile_id = profile.id.trim();
            if profile_id.is_empty() {
                return Err(LlmRuntimeError::InvalidProfileId);
            }
            if !seen.insert(profile_id.to_string()) {
                return Err(LlmRuntimeError::DuplicateProfileId(profile_id.to_string()));
            }

            let mut normalized = profile.clone();
            normalized.id = profile_id.to_string();
            normalized.worker_count = normalized.worker_count.max(1);
            profile_ids.insert(normalized.id.clone());
            profiles.push(normalized);
        }

        if profiles.is_empty() {
            return Err(LlmRuntimeError::MissingProfiles);
        }

        for (route_name, profile_id) in [
            (
                "turn_generation_profile",
                &self.routing.turn_generation_profile,
            ),
            (
                "fact_extraction_profile",
                &self.routing.fact_extraction_profile,
            ),
        ] {
            if !profile_ids.contains(profile_id) {
                return Err(LlmRuntimeError::MissingRoutedProfile {
                    route: route_name,
                    profile_id: profile_id.clone(),
                });
            }
        }

        Ok(ValidatedLlmRuntimeConfig {
            profiles,
            routing: self.routing.clone(),
        })
    }

    pub fn profiles_by_id(
        &self,
    ) -> Result<BTreeMap<String, LlmRuntimeProfileConfig>, LlmRuntimeError> {
        let validated = self.validate()?;
        Ok(validated
            .profiles
            .into_iter()
            .map(|profile| (profile.id.clone(), profile))
            .collect())
    }
}

/// Errors produced by the runtime.
#[derive(Debug, thiserror::Error)]
pub enum LlmRuntimeError {
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("serde json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("http error: {0}")]
    Http(#[from] ureq::Error),
    #[error("model weights were not found in the repository cache")]
    MissingWeights,
    #[error("unsupported model configuration: {0}")]
    UnsupportedModel(&'static str),
    #[error("runtime config must contain at least one profile")]
    MissingProfiles,
    #[error("runtime profile ids must be non-empty")]
    InvalidProfileId,
    #[error("duplicate runtime profile id: {0}")]
    DuplicateProfileId(String),
    #[error("routed profile `{profile_id}` for `{route}` was not configured")]
    MissingRoutedProfile {
        route: &'static str,
        profile_id: String,
    },
    #[error("the model config did not specify an end-of-sequence token")]
    MissingEosToken,
    #[error("failed to parse llm turn: {0}")]
    InvalidTurn(String),
    #[error("generation cancelled")]
    Cancelled,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

/// A tool call emitted by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmToolCall {
    /// Tool name requested by the model.
    pub tool: String,
    /// JSON arguments for the tool.
    pub arguments: Value,
}

/// A structured response from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmTurn {
    /// Agent identifier associated with the turn.
    pub agent: String,
    /// Optional conversation thread identifier associated with the turn.
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// Text response from the model.
    pub response: String,
    /// Optional tool calls captured from the model.
    #[serde(default)]
    pub tool_calls: Vec<LlmToolCall>,
}

/// Stored turn history for a named conversation thread.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmConversationState {
    #[serde(default)]
    pub facts: Vec<LlmConversationFact>,
    #[serde(default)]
    pub entries: Vec<LlmConversationEntry>,
    #[serde(default)]
    pub processed_entry_count: usize,
    #[serde(default)]
    pub next_entry_sequence: u64,
}

/// A structured conversation entry stored by the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmConversationEntry {
    pub speaker: String,
    pub text: String,
    pub kind: LlmConversationEntryKind,
    pub extractable: bool,
    #[serde(default)]
    pub sequence: u64,
}

/// Entry categories used for prompt rendering and fact extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmConversationEntryKind {
    Chat,
    Control,
    Prompt,
    Response,
    ToolCall,
}

/// A durable fact extracted from conversation history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmConversationFact {
    pub fact: String,
    pub importance: f32,
    pub expiry: String,
}

#[derive(Debug, Deserialize)]
struct LlmTurnJson {
    #[serde(default)]
    response: Option<String>,
    #[serde(default)]
    tool_calls: Vec<LlmToolCallJson>,
}

#[derive(Debug, Deserialize)]
struct LlmToolCallJson {
    tool: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Debug, Deserialize)]
struct LlmFactExtractionJson {
    #[serde(default)]
    facts: Vec<LlmConversationFact>,
}

/// A single-model local runtime built on Candle.
pub struct LlmRuntime {
    model: LoadedModel,
    tokenizer: Tokenizer,
    device: Device,
    eos_token: Option<u32>,
    im_end_token: Option<u32>,
    seed: u64,
    temperature: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    max_new_tokens: usize,
    profile_id: String,
    worker_index: usize,
}

enum LoadedModel {
    Dense {
        model: Llama,
        model_config: LlamaRuntimeConfig,
        dtype: DType,
    },
    QuantizedLlama {
        model: QuantizedLlama,
    },
    QuantizedQwen2 {
        model: QuantizedQwen2,
    },
}

impl LlmRuntime {
    /// Load the supported model from Hugging Face and warm the local cache.
    pub fn load(config: &LlmRuntimeProfileConfig) -> Result<Self, LlmRuntimeError> {
        let cache_root = cache_root(config.cache_dir.clone());
        let tokenizer_path = ensure_repo_file(
            &cache_root,
            config.model.tokenizer_repo_id(),
            config.model.revision(),
            "tokenizer.json",
        )?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        let im_end_token = tokenizer.get_vocab(true).get("<|im_end|>").copied();

        let device = choose_device(config.use_gpu)?;
        let (model, eos_token) = match config.model.source_kind() {
            ModelSourceKind::DenseSafetensors => {
                let config_path = ensure_repo_file(
                    &cache_root,
                    config.model.repo_id(),
                    config.model.revision(),
                    "config.json",
                )?;
                let config_json = fs::read_to_string(config_path)?;
                let llama_hf_config: LlamaConfig = serde_json::from_str(&config_json)?;
                let model_config = llama_hf_config.into_config(false);
                let dtype = if matches!(device, Device::Cpu) {
                    DType::F32
                } else {
                    DType::BF16
                };

                let revision_cache = repo_revision_cache(
                    &cache_root,
                    config.model.repo_id(),
                    config.model.revision(),
                )?;
                let weight_files = resolve_weight_files(
                    config.model.repo_id(),
                    config.model.revision(),
                    &revision_cache,
                )?;
                if weight_files.is_empty() {
                    return Err(LlmRuntimeError::MissingWeights);
                }

                let vb =
                    unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
                let model = Llama::load(vb, &model_config)?;
                let eos_token =
                    eos_token_id(&model_config).ok_or(LlmRuntimeError::MissingEosToken)?;
                (
                    LoadedModel::Dense {
                        model,
                        model_config,
                        dtype,
                    },
                    eos_token,
                )
            }
            ModelSourceKind::QuantizedLlamaGguf { gguf_filename } => {
                let gguf_path = ensure_repo_file(
                    &cache_root,
                    config.model.repo_id(),
                    config.model.revision(),
                    gguf_filename,
                )?;
                let mut file = fs::File::open(&gguf_path)?;
                let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
                let model = match config.model {
                    LlmModel::Qwen2_5_1_5BInstructQ2K | LlmModel::Qwen2_5_1_5BInstructQ4KM => {
                        LoadedModel::QuantizedQwen2 {
                            model: QuantizedQwen2::from_gguf(content, &mut file, &device)?,
                        }
                    }
                    _ => LoadedModel::QuantizedLlama {
                        model: QuantizedLlama::from_gguf(content, &mut file, &device)?,
                    },
                };
                let eos_token = tokenizer
                    .get_vocab(true)
                    .get("<|im_end|>")
                    .copied()
                    .or_else(|| tokenizer.get_vocab(true).get("</s>").copied())
                    .or_else(|| tokenizer.get_vocab(true).get("<|endoftext|>").copied())
                    .ok_or(LlmRuntimeError::MissingEosToken)?;
                (model, eos_token)
            }
            ModelSourceKind::Unsupported { reason } => {
                return Err(LlmRuntimeError::UnsupportedModel(reason));
            }
        };

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token: Some(eos_token),
            im_end_token,
            seed: config.seed,
            temperature: config.temperature,
            top_p: config.top_p,
            repeat_penalty: config.repeat_penalty,
            repeat_last_n: config.repeat_last_n,
            max_new_tokens: config.max_new_tokens,
            profile_id: config.id.clone(),
            worker_index: 0,
        })
    }

    pub(crate) fn with_worker_index(mut self, worker_index: usize) -> Self {
        self.worker_index = worker_index;
        self
    }

    fn worker_tag(&self) -> String {
        format!("{}:{}", self.profile_id, self.worker_index)
    }

    pub fn refresh_conversation_facts(
        &mut self,
        conversation: &mut LlmConversationState,
        world: Option<&LlmWorldContext>,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<(), LlmRuntimeError> {
        expire_conversation_facts(conversation);
        let Some(extraction_input) = pending_fact_extraction_input(conversation) else {
            trim_conversation_context(conversation);
            return Ok(());
        };

        let facts = self.extract_facts_from_input(&extraction_input, world, cancel_flag)?;
        merge_conversation_facts(&mut conversation.facts, facts);
        conversation.processed_entry_count = conversation.entries.len();
        trim_conversation_context(conversation);
        Ok(())
    }

    /// Extract facts from a raw prompt without mutating any conversation state.
    pub fn extract_facts_from_prompt(
        &mut self,
        prompt: &str,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<Vec<LlmConversationFact>, LlmRuntimeError> {
        self.extract_facts_from_input(prompt, None, cancel_flag)
    }

    /// Generate text for a prompt using the loaded model.
    pub fn generate(&mut self, prompt: &str) -> Result<String, LlmRuntimeError> {
        let prompt = self.format_chat_prompt(prompt);
        let response = self.generate_raw(&prompt, None)?;
        Ok(response)
    }

    fn generate_raw(
        &mut self,
        prompt: &str,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<String, LlmRuntimeError> {
        self.generate_raw_with_stop_tokens(prompt, cancel_flag, &[])
    }

    fn generate_raw_with_stop_tokens(
        &mut self,
        prompt: &str,
        cancel_flag: Option<&AtomicBool>,
        stop_tokens: &[u32],
    ) -> Result<String, LlmRuntimeError> {
        debug!("llm worker {} raw generation started", self.worker_tag());
        check_cancelled(cancel_flag)?;
        let encoding = self.tokenizer.encode(prompt.to_string(), true)?;
        let mut tokens = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            debug!(
                "llm worker {} raw generation produced empty token stream",
                self.worker_tag()
            );
            return Ok(String::new());
        }
        let prompt_len = tokens.len();

        let mut logits_processor = LogitsProcessor::new(self.seed, self.temperature, self.top_p);
        let mut pos = 0;
        let started = Instant::now();
        let worker_tag = self.worker_tag();
        let mut stop_reason = String::from("max_new_tokens");
        let eos_token = self.eos_token;
        let tokenizer = self.tokenizer.clone();
        match &mut self.model {
            LoadedModel::Dense {
                model,
                model_config,
                dtype,
            } => {
                let mut cache = Cache::new(true, *dtype, model_config, &self.device)?;
                for index in 0..self.max_new_tokens {
                    check_cancelled(cancel_flag)?;
                    let context_size = if index > 0 { 1 } else { tokens.len() };
                    let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                    let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                    let logits = model.forward(&input, pos, &mut cache)?.squeeze(0)?;
                    let logits = if self.repeat_penalty == 1.0 {
                        logits
                    } else {
                        let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                        apply_repeat_penalty(&logits, self.repeat_penalty, &tokens[start_at..])?
                    };
                    let next_token = logits_processor.sample(&logits)?;
                    tokens.push(next_token);
                    let token_text = self
                        .tokenizer
                        .decode(&[next_token], false)
                        .unwrap_or_else(|_| String::from("<decode-error>"));
                    debug!(
                        "llm worker {} generated token {}/{} id={} text={:?}",
                        worker_tag,
                        index + 1,
                        self.max_new_tokens,
                        next_token,
                        token_text
                    );
                    if let Some(reason) =
                        should_stop_on_token(&tokenizer, eos_token, next_token, stop_tokens)
                    {
                        stop_reason = reason;
                        break;
                    }
                    pos += context_size;
                }
            }
            LoadedModel::QuantizedLlama { model } => {
                let mut model = model.clone();
                for index in 0..self.max_new_tokens {
                    check_cancelled(cancel_flag)?;
                    let context_size = if index > 0 { 1 } else { tokens.len() };
                    let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                    let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                    let logits = model.forward(&input, pos)?.squeeze(0)?;
                    let logits = if self.repeat_penalty == 1.0 {
                        logits
                    } else {
                        let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                        apply_repeat_penalty(&logits, self.repeat_penalty, &tokens[start_at..])?
                    };
                    let next_token = logits_processor.sample(&logits)?;
                    tokens.push(next_token);
                    let token_text = self
                        .tokenizer
                        .decode(&[next_token], false)
                        .unwrap_or_else(|_| String::from("<decode-error>"));
                    debug!(
                        "llm worker {} generated token {}/{} id={} text={:?}",
                        worker_tag,
                        index + 1,
                        self.max_new_tokens,
                        next_token,
                        token_text
                    );
                    if let Some(reason) =
                        should_stop_on_token(&tokenizer, eos_token, next_token, stop_tokens)
                    {
                        stop_reason = reason;
                        break;
                    }
                    pos += context_size;
                }
            }
            LoadedModel::QuantizedQwen2 { model } => {
                for index in 0..self.max_new_tokens {
                    check_cancelled(cancel_flag)?;
                    let context_size = if index > 0 { 1 } else { tokens.len() };
                    let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                    let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                    let logits = model.forward(&input, pos)?.squeeze(0)?;
                    let logits = if self.repeat_penalty == 1.0 {
                        logits
                    } else {
                        let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                        apply_repeat_penalty(&logits, self.repeat_penalty, &tokens[start_at..])?
                    };
                    let next_token = logits_processor.sample(&logits)?;
                    tokens.push(next_token);
                    let token_text = self
                        .tokenizer
                        .decode(&[next_token], false)
                        .unwrap_or_else(|_| String::from("<decode-error>"));
                    debug!(
                        "llm worker {} generated token {}/{} id={} text={:?}",
                        worker_tag,
                        index + 1,
                        self.max_new_tokens,
                        next_token,
                        token_text
                    );
                    if let Some(reason) =
                        should_stop_on_token(&tokenizer, eos_token, next_token, stop_tokens)
                    {
                        stop_reason = reason;
                        break;
                    }
                    pos += context_size;
                }
            }
        }

        let generated = &tokens[prompt_len..];
        let generated = if generated
            .last()
            .is_some_and(|token| Some(*token) == self.eos_token || stop_tokens.contains(token))
        {
            &generated[..generated.len().saturating_sub(1)]
        } else {
            generated
        };
        let decoded = self.tokenizer.decode(generated, true)?;
        let raw_decoded = self
            .tokenizer
            .decode(generated, false)
            .unwrap_or_else(|_| String::from("<decode-error>"));
        debug!(
            "llm worker {} raw generation finished in {:?}; stop_reason={}; visible_output={}; raw_output={:?}",
            self.worker_tag(),
            started.elapsed(),
            stop_reason,
            decoded,
            raw_decoded
        );
        Ok(decoded)
    }

    fn extract_facts_from_input(
        &mut self,
        extraction_input: &str,
        world: Option<&LlmWorldContext>,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<Vec<LlmConversationFact>, LlmRuntimeError> {
        let started = Instant::now();
        let prompt = self.build_fact_extraction_prompt(extraction_input, world);
        info!(
            "llm worker {} fact extraction prompt:\n{}",
            self.worker_tag(),
            prompt
        );
        let response = self.generate_raw(&prompt, cancel_flag)?;
        let facts = parse_fact_extraction_response(&response);
        info!(
            "llm worker {} fact extraction finished in {:?}; new_facts={}",
            self.worker_tag(),
            started.elapsed(),
            facts.len()
        );
        for fact in &facts {
            info!(
                "llm worker {} fact -> importance={:.2} expiry={} fact={}",
                self.worker_tag(),
                fact.importance,
                fact.expiry,
                fact.fact
            );
        }
        Ok(facts)
    }

    /// Generate a structured turn for a specific agent.
    pub fn generate_turn(
        &mut self,
        agent: &str,
        conversation_id: Option<&str>,
        prompt: &str,
        response_mode: LlmResponseMode,
        cancel_flag: Option<&AtomicBool>,
        facts: &[LlmConversationFact],
        tools: &[ToolDefinition],
        world: Option<&LlmWorldContext>,
        conversation: Option<&LlmConversationState>,
        session_participants: Option<&[String]>,
        speaker_labels: Option<&HashMap<String, String>>,
    ) -> Result<LlmTurn, LlmRuntimeError> {
        let prompt = self.build_turn_prompt(
            agent,
            prompt,
            response_mode,
            facts,
            tools,
            world,
            conversation,
            session_participants,
            speaker_labels,
        );
        let started = Instant::now();
        info!(
            "llm worker {} turn prompt: agent={} conversation={:?} mode={:?}\n{}",
            self.worker_tag(),
            agent,
            conversation_id,
            response_mode,
            prompt
        );
        let response = match response_mode {
            LlmResponseMode::PlainTextChat => {
                let mut stop_tokens = Vec::new();
                if let Some(im_end_token) = self.im_end_token {
                    stop_tokens.push(im_end_token);
                }
                self.generate_raw_with_stop_tokens(&prompt, cancel_flag, &stop_tokens)?
            }
            LlmResponseMode::StructuredJson => self.generate_raw(&prompt, cancel_flag)?,
        };
        let turn = self.parse_turn(agent, conversation_id, response_mode, world, &response)?;
        debug!(
            "llm worker {} turn generation finished in {:?}; agent={} conversation={:?} mode={:?}",
            self.worker_tag(),
            started.elapsed(),
            agent,
            conversation_id,
            response_mode
        );
        let turn = normalize_and_filter_tool_calls(turn, response_mode, tools);
        match response_mode {
            LlmResponseMode::PlainTextChat => {
                if turn.response == "SILENT" {
                    info!(
                        "llm worker {} chat turn silent: agent={} conversation={:?}",
                        self.worker_tag(),
                        agent,
                        conversation_id
                    );
                } else if !turn.response.is_empty() {
                    info!(
                        "llm worker {} chat turn: agent={} conversation={:?} text={}",
                        self.worker_tag(),
                        agent,
                        conversation_id,
                        turn.response
                    );
                }
            }
            LlmResponseMode::StructuredJson => {
                if !turn.response.is_empty() {
                    info!(
                        "llm worker {} response: agent={} conversation={:?} text={}",
                        self.worker_tag(),
                        agent,
                        conversation_id,
                        turn.response
                    );
                }
            }
        }
        Ok(turn)
    }

    fn build_turn_prompt(
        &self,
        agent: &str,
        prompt: &str,
        response_mode: LlmResponseMode,
        facts: &[LlmConversationFact],
        tools: &[ToolDefinition],
        world: Option<&LlmWorldContext>,
        conversation: Option<&LlmConversationState>,
        session_participants: Option<&[String]>,
        speaker_labels: Option<&HashMap<String, String>>,
    ) -> String {
        let (response_contract, tool_block) = match response_mode {
            LlmResponseMode::StructuredJson => {
                let tool_block = if tools.is_empty() {
                    String::from("No tools are available for this request.")
                } else {
                    let mut lines = Vec::with_capacity(tools.len());
                    for tool in tools {
                        let schema = serde_json::to_string(&tool.parameters)
                            .unwrap_or_else(|_| String::from("{}"));
                        lines.push(format!(
                            "- tool: {}\n  description: {}\n  arguments schema: {}",
                            tool.name, tool.description, schema
                        ));
                    }
                    if tools.len() == 1 {
                        format!(
                            "Available tools:\n{}\nOnly valid tool name for this request: {}",
                            lines.join("\n"),
                            tools[0].name
                        )
                    } else {
                        format!("Available tools:\n{}", lines.join("\n"))
                    }
                };

                let response_contract = if tools.is_empty() {
                    String::from(
                        "Return exactly one JSON object.\n\
No extra text.\n\
No tools are available for this request.\n\
Respond with a JSON object containing only `response`.\n\
Never return `tool_calls`.\n",
                    )
                } else {
                    String::from(
                        "Return exactly one JSON object.\n\
No extra text.\n\
If calling tools, respond with a JSON object containing only a `tool_calls` array.\n\
Each entry must have `tool` and `arguments` fields.\n\
If not calling tools, respond with a JSON object containing only `response`.\n\
Never include both fields.\n\
Use exact tool names only.\n",
                    )
                };

                (response_contract, tool_block)
            }
            LlmResponseMode::PlainTextChat => (
                build_plain_text_chat_response_preamble(agent, world, session_participants),
                String::from("Tools are disabled for this local chat request."),
            ),
        };

        let world_block = match world {
            Some(world) => format!(
                "World state:\n{}\n\nWorld fields:\n{}",
                format_json(&world.world_view),
                format_schema_guide(&world.world_schema),
            ),
            None => String::from("No world state snapshot is available for this agent."),
        };

        let conversation_block = match conversation {
            Some(conversation) => format_conversation_context(conversation),
            None => String::from("No conversation history is active for this request."),
        };

        let fact_block = format_fact_memory(facts);

        let content = match response_mode {
            LlmResponseMode::StructuredJson => format!(
                "Agent: {agent}\n\
{response_contract}\
World context:\n\
{world_block}\n\
Memory facts:\n\
{fact_block}\n\
Conversation context:\n\
{conversation_block}\n\
{tool_block}\n\
Game prompt:\n\
{prompt}\n\
",
                response_contract = response_contract,
                world_block = world_block,
                fact_block = fact_block,
                conversation_block = conversation_block,
                tool_block = tool_block,
            ),
            LlmResponseMode::PlainTextChat => {
                return self.build_plain_text_chat_prompt(
                    agent,
                    &response_contract,
                    &fact_block,
                    conversation,
                    prompt,
                    world,
                    session_participants,
                    speaker_labels,
                );
            }
        };
        self.format_chat_prompt(&content)
    }

    fn build_plain_text_chat_prompt(
        &self,
        agent: &str,
        response_contract: &str,
        _fact_block: &str,
        conversation: Option<&LlmConversationState>,
        prompt: &str,
        world: Option<&LlmWorldContext>,
        session_participants: Option<&[String]>,
        speaker_labels: Option<&HashMap<String, String>>,
    ) -> String {
        let speaker_label = plain_text_chat_speaker_label(agent, world);
        let mut prompt_sections = vec![chatml_message(
            "system",
            &format!(
                "{response_contract}Tools are disabled for this local chat request.\nCurrent prompt:\n{prompt}\n\nOutput contract:\nReply with message text only.\nDo not include your name, speaker label, markdown, or surrounding quotes.\nIf your reply would only repeat memory facts or restate something already said, reply exactly `SILENT`.\nIf you have nothing useful to say, reply exactly `SILENT`.\nAccepted legacy format: `{}`.\n",
                expected_plain_text_chat_response_format(agent, world),
            ),
        )];

        if let Some(participants) = session_participants {
            if !participants.is_empty() {
                prompt_sections.push(chatml_message(
                    "system",
                    &format!("Nearby people: {}", participants.join(", ")),
                ));
            }
        }

        if let Some(conversation) = conversation {
            for entry in &conversation.entries {
                prompt_sections.push(render_plain_text_chat_entry(
                    agent,
                    entry,
                    world,
                    speaker_labels,
                ));
            }
        }

        prompt_sections.push(format!(
            "<|im_start|>assistant\n{}",
            plain_text_chat_opening_cue(&speaker_label)
        ));
        prompt_sections.join("\n")
    }

    fn format_chat_prompt(&self, user_content: &str) -> String {
        format!("<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n")
    }

    fn build_fact_extraction_prompt(
        &self,
        extraction_input: &str,
        world: Option<&LlmWorldContext>,
    ) -> String {
        let content = build_fact_extraction_prompt_content(extraction_input, world);
        self.format_chat_prompt(&content)
    }

    fn parse_turn(
        &self,
        agent: &str,
        conversation_id: Option<&str>,
        response_mode: LlmResponseMode,
        world: Option<&LlmWorldContext>,
        output: &str,
    ) -> Result<LlmTurn, LlmRuntimeError> {
        let cleaned = strip_code_fences(output).trim().to_string();
        debug!(
            "llm worker {} parsed candidate output: {}",
            self.worker_tag(),
            cleaned
        );

        if response_mode == LlmResponseMode::PlainTextChat {
            return Ok(LlmTurn {
                agent: agent.to_string(),
                conversation_id: conversation_id.map(str::to_string),
                response: normalize_plain_text_chat_response(agent, world, &cleaned),
                tool_calls: Vec::new(),
            });
        }

        let json_objects = extract_json_objects(&cleaned);
        if !json_objects.is_empty() {
            let mut response = None;
            let mut tool_calls = Vec::new();

            for candidate in json_objects {
                if let Ok(json) = serde_json::from_str::<LlmTurnJson>(candidate) {
                    let has_turn_fields = json.response.is_some() || !json.tool_calls.is_empty();
                    if has_turn_fields {
                        if response.is_none() {
                            response = json.response;
                        }
                        tool_calls.extend(json.tool_calls.into_iter().map(|call| LlmToolCall {
                            tool: call.tool,
                            arguments: call.arguments,
                        }));
                        continue;
                    }
                }

                if let Ok(call) = serde_json::from_str::<LlmToolCallJson>(candidate) {
                    tool_calls.push(LlmToolCall {
                        tool: call.tool,
                        arguments: call.arguments,
                    });
                }
            }

            if response.is_some() || !tool_calls.is_empty() {
                let response = if tool_calls.is_empty() {
                    response.unwrap_or_default()
                } else {
                    String::new()
                };
                debug!(
                    "llm worker {} parsed structured turn: response_len={} tool_calls={}",
                    self.worker_tag(),
                    response.len(),
                    tool_calls.len()
                );
                return Ok(LlmTurn {
                    agent: agent.to_string(),
                    conversation_id: conversation_id.map(str::to_string),
                    response,
                    tool_calls,
                });
            }
        }

        Ok(LlmTurn {
            agent: agent.to_string(),
            conversation_id: conversation_id.map(str::to_string),
            response: cleaned,
            tool_calls: Vec::new(),
        })
    }
}

fn build_plain_text_chat_identity_preamble(
    _agent: &str,
    world: Option<&LlmWorldContext>,
) -> String {
    let agent_name = world_view_string_field(world, "agent_name");
    let persona = world_view_string_field(world, "persona");
    let current_goal = world_view_string_field(world, "current_goal");

    let mut lines = Vec::new();
    if let Some(agent_name) = agent_name {
        lines.push(format!("Preferred agent name: {agent_name}"));
    }
    if let Some(persona) = persona {
        lines.push(format!("Persona: {persona}"));
    }
    if let Some(current_goal) = current_goal {
        lines.push(format!("Current goal: {current_goal}"));
    }
    if lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", lines.join("\n"))
    }
}

fn build_plain_text_chat_response_preamble(
    agent: &str,
    world: Option<&LlmWorldContext>,
    session_participants: Option<&[String]>,
) -> String {
    let _ = session_participants;
    build_plain_text_chat_identity_preamble(agent, world)
}

fn chatml_message(role: &str, content: &str) -> String {
    format!("<|im_start|>{role}\n{content}<|im_end|>")
}

fn plain_text_chat_speaker_label(agent: &str, world: Option<&LlmWorldContext>) -> String {
    world_view_string_field(world, "agent_name")
        .unwrap_or(agent)
        .to_string()
}

fn render_plain_text_chat_entry(
    active_agent: &str,
    entry: &LlmConversationEntry,
    active_world: Option<&LlmWorldContext>,
    speaker_labels: Option<&HashMap<String, String>>,
) -> String {
    match entry.kind {
        LlmConversationEntryKind::Control => chatml_message("system", &entry.text),
        LlmConversationEntryKind::Chat
        | LlmConversationEntryKind::Prompt
        | LlmConversationEntryKind::Response
        | LlmConversationEntryKind::ToolCall => {
            let speaker_label = if entry.speaker == active_agent {
                plain_text_chat_speaker_label(active_agent, active_world)
            } else {
                speaker_labels
                    .and_then(|labels| labels.get(&entry.speaker))
                    .cloned()
                    .unwrap_or_else(|| entry.speaker.clone())
            };
            let role = if entry.speaker == active_agent {
                "assistant"
            } else {
                "user"
            };
            chatml_message(role, &format!("**{}:** {}", speaker_label, entry.text))
        }
    }
}

fn build_fact_extraction_prompt_content(
    extraction_input: &str,
    _world: Option<&LlmWorldContext>,
) -> String {
    format!(
        "Extract durable facts from the following conversation updates.\n\
Return exactly one JSON object with a single `facts` array.\n\
Each fact entry must contain `fact`, `importance`, and `expiry`.\n\
`importance` must be a number between 0.0 and 1.0.\n\
Use short concrete facts only.\n\
Only extract facts that were explicitly stated in the conversation updates.\n\
If nothing should be remembered, return {{\"facts\":[]}}.\n\
Do not omit the `fact` field.\n\
Valid example with one fact:\n\
{{\"facts\":[{{\"fact\":\"Alpha ordered the bridge held.\",\"importance\":0.85,\"expiry\":\"unknown\"}}]}}\n\
Valid example with no facts:\n\
{{\"facts\":[]}}\n\
Do not include any text outside the JSON object.\n\
\n\
Conversation updates:\n\
{extraction_input}\n"
    )
}

fn expected_plain_text_chat_response_format(
    agent: &str,
    world: Option<&LlmWorldContext>,
) -> String {
    let speaker = world_view_string_field(world, "agent_name").unwrap_or(agent);
    format!("**{speaker}:** {{message}}")
}

fn plain_text_chat_opening_cue(speaker_label: &str) -> String {
    format!(
        "Respond as {speaker_label}. Continue the conversation with message text only. If everything important has already been covered, reply with SILENT.\n"
    )
}

fn normalize_plain_text_chat_response(
    agent: &str,
    world: Option<&LlmWorldContext>,
    output: &str,
) -> String {
    if output == "SILENT" {
        return String::from("SILENT");
    }

    let mut accepted_labels = vec![agent];
    if let Some(agent_name) = world_view_string_field(world, "agent_name") {
        if agent_name != agent {
            accepted_labels.push(agent_name);
        }
    }

    if let Some(message) = strip_plain_text_chat_speaker_prefix(output, &accepted_labels) {
        return message;
    }

    if output.trim() == "**" {
        return String::from("SILENT");
    }

    output.to_string()
}

fn should_stop_on_token(
    tokenizer: &Tokenizer,
    eos_token: Option<u32>,
    token: u32,
    stop_tokens: &[u32],
) -> Option<String> {
    if Some(token) == eos_token {
        return Some(format!("eos_token({token})"));
    }
    if stop_tokens.contains(&token) {
        return Some(format!("stop_token({token})"));
    }

    let decoded = tokenizer.decode(&[token], false).ok()?;
    if decoded.is_empty() {
        return Some(format!("empty_special_token({token})"));
    }

    None
}

fn strip_plain_text_chat_speaker_prefix(output: &str, accepted_labels: &[&str]) -> Option<String> {
    let content = output.strip_prefix("**")?;
    for separator in ["**:", ":**"] {
        let Some((label, message)) = content.split_once(separator) else {
            continue;
        };
        if accepted_labels
            .iter()
            .any(|candidate| *candidate == label.trim())
        {
            return Some(message.trim().to_string());
        }
    }
    None
}

fn world_view_string_field<'a>(world: Option<&'a LlmWorldContext>, key: &str) -> Option<&'a str> {
    world
        .and_then(|world| world.world_view.as_object())
        .and_then(|view| view.get(key))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

fn choose_device(use_gpu: bool) -> Result<Device, candle_core::Error> {
    if use_gpu && candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if use_gpu && candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

fn resolve_weight_files(
    repo_id: &str,
    revision: &str,
    revision_cache: &Path,
) -> Result<Vec<PathBuf>, LlmRuntimeError> {
    let index_path = ensure_file(
        repo_id,
        revision,
        "model.safetensors.index.json",
        revision_cache,
    );
    if let Ok(index_path) = index_path {
        let index_json = fs::read_to_string(index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&index_json)?;
        let files = index
            .weight_map
            .into_values()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|filename| ensure_file(repo_id, revision, &filename, revision_cache))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(files);
    }

    if let Ok(single_file) = ensure_file(repo_id, revision, "model.safetensors", revision_cache) {
        return Ok(vec![single_file]);
    }

    Ok(Vec::new())
}

fn ensure_repo_file(
    cache_root: &Path,
    repo_id: &str,
    revision: &str,
    filename: &str,
) -> Result<PathBuf, LlmRuntimeError> {
    let revision_cache = repo_revision_cache(cache_root, repo_id, revision)?;
    ensure_file(repo_id, revision, filename, &revision_cache)
}

fn repo_revision_cache(
    cache_root: &Path,
    repo_id: &str,
    revision: &str,
) -> Result<PathBuf, LlmRuntimeError> {
    let revision_cache = cache_root.join(safe_repo_id(repo_id)).join(revision);
    fs::create_dir_all(&revision_cache)?;
    Ok(revision_cache)
}

fn ensure_file(
    repo_id: &str,
    revision: &str,
    filename: &str,
    revision_cache: &Path,
) -> Result<PathBuf, LlmRuntimeError> {
    let local_path = revision_cache.join(filename);
    if local_path.exists() {
        return Ok(local_path);
    }

    let url = format!("https://huggingface.co/{repo_id}/resolve/{revision}/{filename}");
    let response = ureq::get(&url).call()?;

    if let Some(parent) = local_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut reader = response.into_reader();
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;

    let mut file = fs::File::create(&local_path)?;
    file.write_all(&bytes)?;
    Ok(local_path)
}

fn cache_root(cache_dir: Option<PathBuf>) -> PathBuf {
    match cache_dir {
        Some(cache_dir) => cache_dir,
        None => {
            let mut root = std::env::var_os("HF_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| dirs_cache_dir().unwrap_or_else(|| PathBuf::from(".cache")));
            root.push("bevellm");
            root.push("hf");
            root
        }
    }
}

fn dirs_cache_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache"))
}

fn safe_repo_id(repo_id: &str) -> String {
    repo_id.replace('/', "__")
}

fn eos_token_id(config: &LlamaRuntimeConfig) -> Option<u32> {
    match &config.eos_token_id {
        Some(LlamaEosToks::Single(token)) => Some(*token),
        Some(LlamaEosToks::Multiple(tokens)) => tokens.first().copied(),
        None => None,
    }
}

fn strip_code_fences(output: &str) -> String {
    let trimmed = output.trim();
    let stripped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    let stripped = stripped.strip_suffix("```").unwrap_or(stripped);
    stripped.trim().to_string()
}

fn extract_json_objects(output: &str) -> Vec<&str> {
    let mut objects = Vec::new();
    let mut depth = 0usize;
    let mut start = None;
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in output.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                if depth == 0 {
                    continue;
                }
                depth -= 1;
                if depth == 0 {
                    if let Some(start_idx) = start.take() {
                        objects.push(&output[start_idx..=idx]);
                    }
                }
            }
            _ => {}
        }
    }

    objects
}

fn normalize_and_filter_tool_calls(
    mut turn: LlmTurn,
    response_mode: LlmResponseMode,
    tools: &[ToolDefinition],
) -> LlmTurn {
    if response_mode == LlmResponseMode::PlainTextChat || tools.is_empty() {
        turn.tool_calls.clear();
        return turn;
    }

    if tools.len() != 1 {
        let allowed = tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<HashSet<_>>();
        turn.tool_calls
            .retain(|call| allowed.contains(call.tool.as_str()));
        return turn;
    }

    let only_tool = tools[0].name.as_str();
    for call in &mut turn.tool_calls {
        if call.tool == "name" || call.tool == "tool" {
            call.tool = only_tool.to_string();
        }
    }
    turn.tool_calls.retain(|call| call.tool == only_tool);

    turn
}

fn format_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

fn format_schema_guide(schema: &Value) -> String {
    let mut lines = Vec::new();
    let mut visited_refs = HashSet::new();
    flatten_schema_descriptions(schema, schema, "", &mut lines, &mut visited_refs);

    if lines.is_empty() {
        String::from("No field descriptions available.")
    } else {
        lines.join("\n")
    }
}

fn flatten_schema_descriptions(
    root: &Value,
    schema: &Value,
    path: &str,
    lines: &mut Vec<String>,
    visited_refs: &mut HashSet<String>,
) {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        if !visited_refs.insert(format!("{path}:{reference}")) {
            return;
        }
        if let Some(resolved) = resolve_schema_ref(root, reference) {
            flatten_schema_descriptions(root, resolved, path, lines, visited_refs);
        }
        return;
    }

    if let Some(description) = schema.get("description").and_then(Value::as_str) {
        if !path.is_empty() {
            lines.push(format!("- {path}: {description}"));
        }
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (name, child) in properties {
            let child_path = join_schema_path(path, name);
            flatten_schema_descriptions(root, child, &child_path, lines, visited_refs);
        }
    }

    if let Some(items) = schema.get("items") {
        let item_path = if path.is_empty() {
            String::from("[]")
        } else {
            format!("{path}[]")
        };

        match items {
            Value::Object(map) => {
                flatten_schema_descriptions(
                    root,
                    &Value::Object(map.clone()),
                    &item_path,
                    lines,
                    visited_refs,
                );
            }
            Value::Array(entries) => {
                for entry in entries {
                    flatten_schema_descriptions(root, entry, &item_path, lines, visited_refs);
                }
            }
            _ => {}
        }
    }
}

fn resolve_schema_ref<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    let pointer = reference.strip_prefix('#')?;
    root.pointer(pointer)
}

fn join_schema_path(prefix: &str, segment: &str) -> String {
    if prefix.is_empty() {
        segment.to_string()
    } else {
        format!("{prefix}.{segment}")
    }
}

fn check_cancelled(cancel_flag: Option<&AtomicBool>) -> Result<(), LlmRuntimeError> {
    if cancel_flag.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        Err(LlmRuntimeError::Cancelled)
    } else {
        Ok(())
    }
}

fn format_conversation_context(conversation: &LlmConversationState) -> String {
    if conversation.entries.is_empty() {
        return String::from("No conversation history is active for this request.");
    }

    conversation
        .entries
        .iter()
        .map(render_conversation_entry)
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_fact_memory(facts: &[LlmConversationFact]) -> String {
    if facts.is_empty() {
        return String::from("No stored facts.");
    }

    facts
        .iter()
        .map(|fact| {
            format!(
                "- fact: {} | importance: {:.2} | expiry: {}",
                fact.fact, fact.importance, fact.expiry
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn append_conversation_turn(
    conversation: &mut LlmConversationState,
    agent: &str,
    prompt: &str,
    response: &str,
    tool_calls: &[LlmToolCall],
) {
    push_entry(
        conversation,
        String::from("user"),
        prompt.to_string(),
        LlmConversationEntryKind::Prompt,
        false,
    );
    if tool_calls.is_empty() {
        push_entry(
            conversation,
            agent.to_string(),
            response.to_string(),
            LlmConversationEntryKind::Response,
            false,
        );
    } else {
        let tool_names = tool_calls
            .iter()
            .map(|call| call.tool.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        push_entry(
            conversation,
            agent.to_string(),
            format!("tool calls: {tool_names}"),
            LlmConversationEntryKind::ToolCall,
            false,
        );
    }
}

pub fn append_conversation_chat(
    conversation: &mut LlmConversationState,
    speaker: impl Into<String>,
    message: impl Into<String>,
    extractable: bool,
) {
    push_entry(
        conversation,
        speaker.into(),
        message.into(),
        LlmConversationEntryKind::Chat,
        extractable,
    );
}

pub fn append_conversation_control(
    conversation: &mut LlmConversationState,
    message: impl Into<String>,
) {
    push_entry(
        conversation,
        String::from("system"),
        message.into(),
        LlmConversationEntryKind::Control,
        false,
    );
}

pub fn compact_conversation_state(conversation: &mut LlmConversationState) {
    expire_conversation_facts(conversation);
    trim_conversation_context(conversation);
    trim_conversation_facts(conversation);
}

fn pending_fact_extraction_input(conversation: &LlmConversationState) -> Option<String> {
    let mut lines = Vec::new();

    for entry in conversation
        .entries
        .iter()
        .skip(conversation.processed_entry_count)
    {
        if entry.extractable && entry.kind == LlmConversationEntryKind::Chat {
            lines.push(render_conversation_entry(entry));
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

fn parse_fact_extraction_response(output: &str) -> Vec<LlmConversationFact> {
    let cleaned = strip_code_fences(output).trim().to_string();
    let mut facts = Vec::new();

    for candidate in extract_json_objects(&cleaned) {
        if let Ok(value) = serde_json::from_str::<Value>(candidate) {
            if value.get("facts").is_some() {
                if let Ok(json) = serde_json::from_value::<LlmFactExtractionJson>(value) {
                    facts.extend(json.facts);
                    continue;
                }
            }
        }

        if let Some(fact) = parse_single_fact_object(candidate) {
            facts.push(fact);
            continue;
        }
    }

    if facts.is_empty() {
        if let Ok(value) = serde_json::from_str::<Value>(&cleaned) {
            if value.get("facts").is_some() {
                if let Ok(json) = serde_json::from_value::<LlmFactExtractionJson>(value) {
                    facts.extend(json.facts);
                }
            } else if let Some(fact) = parse_single_fact_object(&cleaned) {
                facts.push(fact);
            }
        } else if let Some(fact) = parse_single_fact_object(&cleaned) {
            facts.push(fact);
        }
    }

    sanitize_facts(facts)
}

fn parse_single_fact_object(output: &str) -> Option<LlmConversationFact> {
    let value = serde_json::from_str::<Value>(output).ok()?;
    let object = value.as_object()?;
    if !object.contains_key("fact") {
        return None;
    }
    serde_json::from_value::<LlmConversationFact>(value).ok()
}

fn sanitize_facts(facts: Vec<LlmConversationFact>) -> Vec<LlmConversationFact> {
    facts
        .into_iter()
        .filter_map(|fact| {
            let text = fact.fact.trim();
            if text.is_empty() {
                return None;
            }

            let expiry = fact.expiry.trim();
            Some(LlmConversationFact {
                fact: text.to_string(),
                importance: fact.importance.clamp(0.0, 1.0),
                expiry: if expiry.is_empty() {
                    String::from("unknown")
                } else {
                    expiry.to_string()
                },
            })
        })
        .collect()
}

pub fn merge_conversation_facts(
    stored_facts: &mut Vec<LlmConversationFact>,
    incoming_facts: Vec<LlmConversationFact>,
) {
    prune_expired_facts(stored_facts);
    for fact in incoming_facts {
        if let Some(existing) = stored_facts
            .iter_mut()
            .find(|existing| existing.fact.eq_ignore_ascii_case(&fact.fact))
        {
            if fact.importance >= existing.importance {
                *existing = fact;
            }
        } else {
            stored_facts.push(fact);
        }
    }

    trim_facts(stored_facts);
}

fn trim_conversation_context(conversation: &mut LlmConversationState) {
    const MAX_RECENT_ENTRIES: usize = 12;

    if conversation.entries.len() > MAX_RECENT_ENTRIES {
        let remove_count = conversation.entries.len() - MAX_RECENT_ENTRIES;
        conversation.entries.drain(..remove_count);
        conversation.processed_entry_count = conversation
            .processed_entry_count
            .saturating_sub(remove_count);
    }
}

fn trim_conversation_facts(conversation: &mut LlmConversationState) {
    trim_facts(&mut conversation.facts);
}

fn trim_facts(facts: &mut Vec<LlmConversationFact>) {
    const MAX_FACTS: usize = 24;

    facts.sort_by(|left, right| {
        right
            .importance
            .partial_cmp(&left.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.fact.cmp(&right.fact))
    });
    if facts.len() > MAX_FACTS {
        facts.truncate(MAX_FACTS);
    }
}

pub fn expire_conversation_facts(conversation: &mut LlmConversationState) {
    prune_expired_facts(&mut conversation.facts);
}

fn push_entry(
    conversation: &mut LlmConversationState,
    speaker: String,
    text: String,
    kind: LlmConversationEntryKind,
    extractable: bool,
) {
    let sequence = conversation.next_entry_sequence;
    conversation.next_entry_sequence = conversation.next_entry_sequence.wrapping_add(1);
    conversation.entries.push(LlmConversationEntry {
        speaker,
        text,
        kind,
        extractable,
        sequence,
    });
}

fn render_conversation_entry(entry: &LlmConversationEntry) -> String {
    match entry.kind {
        LlmConversationEntryKind::Chat => format!("**{}**: {}", entry.speaker, entry.text),
        LlmConversationEntryKind::Control => format!("[system] {}", entry.text),
        LlmConversationEntryKind::Prompt => format!("[prompt] {}", entry.text),
        LlmConversationEntryKind::Response => {
            format!("[response:{}] {}", entry.speaker, entry.text)
        }
        LlmConversationEntryKind::ToolCall => {
            format!("[tools:{}] {}", entry.speaker, entry.text)
        }
    }
}

fn prune_expired_facts(facts: &mut Vec<LlmConversationFact>) {
    facts.retain(|fact| !fact_is_expired(fact));
}

fn fact_is_expired(fact: &LlmConversationFact) -> bool {
    let expiry = fact.expiry.trim();
    if expiry.eq_ignore_ascii_case("unknown") {
        return false;
    }

    let Some(expiry_days) = parse_expiry_days(expiry) else {
        return true;
    };
    let Some(today_days) = current_utc_days_since_epoch() else {
        return true;
    };
    expiry_days < today_days
}

fn parse_expiry_days(expiry: &str) -> Option<i64> {
    let date = expiry.get(..10)?;
    let mut parts = date.split('-');
    let year = parts.next()?.parse::<i32>().ok()?;
    let month = parts.next()?.parse::<u32>().ok()?;
    let day = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    days_from_civil(year, month, day)
}

fn current_utc_days_since_epoch() -> Option<i64> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    Some((now.as_secs() / 86_400) as i64)
}

fn days_from_civil(year: i32, month: u32, day: u32) -> Option<i64> {
    if !(1..=12).contains(&month) || day == 0 {
        return None;
    }
    let year = year - i32::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = month as i32;
    let day = day as i32;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let leap = yoe / 4 - yoe / 100 + yoe / 400;
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap > 0 => 29,
        2 => 28,
        _ => return None,
    };
    if day > max_day {
        return None;
    }
    let doe = yoe * 365 + leap + doy;
    Some((era * 146097 + doe - 719468) as i64)
}

#[cfg(test)]
mod tests {
    use super::{
        LlmConversationEntry, LlmConversationEntryKind, LlmConversationFact, LlmConversationState,
        LlmRuntimeConfig, LlmRuntimeProfileConfig, LlmTaskRoutingConfig, append_conversation_chat,
        append_conversation_control, append_conversation_turn,
        build_fact_extraction_prompt_content, build_plain_text_chat_identity_preamble,
        build_plain_text_chat_response_preamble, compact_conversation_state,
        expected_plain_text_chat_response_format, expire_conversation_facts,
        format_conversation_context, normalize_plain_text_chat_response,
        parse_fact_extraction_response, pending_fact_extraction_input,
        render_plain_text_chat_entry,
    };
    use crate::LlmWorldContext;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn prompt_renderer_formats_chat_and_control_entries() {
        let mut conversation = LlmConversationState::default();
        append_conversation_chat(&mut conversation, "agent_bravo", "example text", true);
        append_conversation_control(&mut conversation, "agent_charlie is now nearby.");

        assert_eq!(
            format_conversation_context(&conversation),
            "**agent_bravo**: example text\n[system] agent_charlie is now nearby."
        );
    }

    #[test]
    fn plain_text_chat_renderer_uses_chatml_roles_by_speaker_perspective() {
        let world = demo_world();
        let speaker_labels =
            HashMap::from([(String::from("agent_bravo"), String::from("Scout Bravo"))]);
        let own_entry = LlmConversationEntry {
            speaker: String::from("agent_alpha"),
            text: String::from("Hold here."),
            kind: LlmConversationEntryKind::Chat,
            extractable: true,
            sequence: 0,
        };
        let other_entry = LlmConversationEntry {
            speaker: String::from("agent_bravo"),
            text: String::from("Scout ahead."),
            kind: LlmConversationEntryKind::Chat,
            extractable: true,
            sequence: 1,
        };

        assert_eq!(
            render_plain_text_chat_entry(
                "agent_alpha",
                &own_entry,
                Some(&world),
                Some(&speaker_labels)
            ),
            "<|im_start|>assistant\n**Commander Alpha:** Hold here.<|im_end|>"
        );
        assert_eq!(
            render_plain_text_chat_entry(
                "agent_alpha",
                &other_entry,
                Some(&world),
                Some(&speaker_labels)
            ),
            "<|im_start|>user\n**Scout Bravo:** Scout ahead.<|im_end|>"
        );
    }

    #[test]
    fn fact_extraction_reads_only_new_extractable_chat_entries() {
        let mut conversation = LlmConversationState::default();
        append_conversation_control(&mut conversation, "session restarted");
        append_conversation_chat(
            &mut conversation,
            "alpha",
            "I hid supplies in the shed.",
            true,
        );
        append_conversation_turn(
            &mut conversation,
            "alpha",
            "Summarize the situation.",
            "We should wait.",
            &[],
        );

        assert_eq!(
            pending_fact_extraction_input(&conversation).as_deref(),
            Some("**alpha**: I hid supplies in the shed.")
        );

        conversation.processed_entry_count = conversation.entries.len();
        append_conversation_chat(&mut conversation, "bravo", "Meet at dusk.", true);
        append_conversation_control(&mut conversation, "charlie is now nearby");

        assert_eq!(
            pending_fact_extraction_input(&conversation).as_deref(),
            Some("**bravo**: Meet at dusk.")
        );
    }

    #[test]
    fn expired_facts_are_removed_but_unknown_expiry_remains() {
        let mut conversation = LlmConversationState {
            facts: vec![
                LlmConversationFact {
                    fact: String::from("stale"),
                    importance: 0.9,
                    expiry: String::from("2000-01-01"),
                },
                LlmConversationFact {
                    fact: String::from("keep"),
                    importance: 0.5,
                    expiry: String::from("unknown"),
                },
                LlmConversationFact {
                    fact: String::from("bad"),
                    importance: 0.4,
                    expiry: String::from("not-a-date"),
                },
            ],
            ..Default::default()
        };

        expire_conversation_facts(&mut conversation);

        assert_eq!(conversation.facts.len(), 1);
        assert_eq!(conversation.facts[0].fact, "keep");
    }

    #[test]
    fn compact_preserves_structured_entry_ordering_metadata() {
        let mut conversation = LlmConversationState::default();
        for idx in 0..16 {
            append_conversation_chat(&mut conversation, "alpha", format!("line {idx}"), true);
        }

        compact_conversation_state(&mut conversation);

        assert_eq!(conversation.entries.len(), 12);
        assert_eq!(conversation.entries[0].kind, LlmConversationEntryKind::Chat);
        assert_eq!(conversation.entries[0].sequence, 4);
    }

    #[test]
    fn plain_text_chat_identity_preamble_includes_name_persona_and_goal() {
        let world = demo_world();

        let preamble = build_plain_text_chat_identity_preamble("agent_alpha", Some(&world));

        assert!(preamble.contains("Preferred agent name: Commander Alpha"));
        assert!(preamble.contains("Persona: Calm field commander who speaks with crisp urgency."));
        assert!(preamble.contains("Current goal: Secure the bridge before the scout advances."));
        assert!(!preamble.contains("Agent id:"));
    }

    #[test]
    fn fact_extraction_prompt_excludes_goals_and_speculation() {
        let prompt = build_fact_extraction_prompt_content("**alpha**: Hold the bridge.", None);

        assert!(prompt.contains("Only extract facts that were explicitly stated"));
        assert!(prompt.contains("\"facts\":[{\"fact\":\"Alpha ordered the bridge held.\""));
        assert!(prompt.contains("Conversation updates:\n**alpha**: Hold the bridge."));
        assert!(!prompt.contains("World context:"));
    }

    #[test]
    fn fact_extraction_parser_accepts_single_fact_object() {
        let facts = parse_fact_extraction_response(
            "{\"fact\":\"Alpha ordered the bridge held.\",\"importance\":0.85,\"expiry\":\"unknown\"}",
        );

        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].fact, "Alpha ordered the bridge held.");
    }

    #[test]
    fn fact_extraction_parser_rejects_partial_fact_objects() {
        let facts = parse_fact_extraction_response(
            "{\n  \"importance\": 0.85,\n  \"expiry\": \"2023-12-31\"\n}",
        );

        assert!(facts.is_empty());
    }

    #[test]
    fn plain_text_chat_prompt_requires_grounded_chat_or_silence() {
        let world = demo_world();

        let prompt = build_plain_text_chat_response_preamble(
            "agent_alpha",
            Some(&world),
            Some(&[String::from("agent_bravo")]),
        );

        assert!(prompt.contains("Preferred agent name: Commander Alpha"));
        assert!(prompt.contains("Persona: Calm field commander who speaks with crisp urgency."));
        assert!(prompt.contains("Current goal: Secure the bridge before the scout advances."));
        assert!(!prompt.contains("Agent id:"));
    }

    #[test]
    fn plain_text_chat_output_contract_uses_expected_name_and_silent() {
        let world = demo_world();

        let speaker_format = expected_plain_text_chat_response_format("agent_alpha", Some(&world));

        assert_eq!(speaker_format, "**Commander Alpha:** {message}");
        assert!(speaker_format.contains("Commander Alpha"));
        assert_eq!(
            normalize_plain_text_chat_response("agent_alpha", Some(&world), "SILENT"),
            "SILENT"
        );
    }

    #[test]
    fn plain_text_chat_output_contract_prefers_message_only_text() {
        let world = demo_world();
        let prompt = format!(
            "{}\nOutput contract:\nReply with message text only.\nDo not include your name, speaker label, markdown, or surrounding quotes.\nIf your reply would only repeat memory facts or restate something already said, reply exactly `SILENT`.\nIf you have nothing useful to say, reply exactly `SILENT`.\nAccepted legacy format: `{}`.\n",
            build_plain_text_chat_response_preamble(
                "agent_alpha",
                Some(&world),
                Some(&[String::from("agent_bravo")]),
            ),
            expected_plain_text_chat_response_format("agent_alpha", Some(&world))
        );

        assert!(prompt.contains("Reply with message text only."));
        assert!(
            prompt.contains(
                "Do not include your name, speaker label, markdown, or surrounding quotes."
            )
        );
        assert!(prompt.contains("If your reply would only repeat memory facts or restate something already said, reply exactly `SILENT`."));
        assert!(prompt.contains("If you have nothing useful to say, reply exactly `SILENT`."));
        assert!(prompt.contains("Accepted legacy format: `**Commander Alpha:** {message}`."));
        assert!(!prompt.contains("Memory facts:"));
    }

    #[test]
    fn plain_text_chat_parser_accepts_agent_id_prefix() {
        let response =
            normalize_plain_text_chat_response("agent_alpha", None, "**agent_alpha**: Move left.");

        assert_eq!(response, "Move left.");
    }

    #[test]
    fn plain_text_chat_parser_accepts_configured_agent_name_prefix() {
        let world = demo_world();
        let response = normalize_plain_text_chat_response(
            "agent_alpha",
            Some(&world),
            "**Commander Alpha**: Move left.",
        );

        assert_eq!(response, "Move left.");
    }

    #[test]
    fn plain_text_chat_parser_treats_bare_bold_marker_as_silent() {
        let world = demo_world();

        assert_eq!(
            normalize_plain_text_chat_response("agent_alpha", Some(&world), "**"),
            "SILENT"
        );
    }

    #[test]
    fn plain_text_chat_parser_accepts_colon_inside_bold_prefix() {
        let world = demo_world();
        let response = normalize_plain_text_chat_response(
            "agent_alpha",
            Some(&world),
            "**Commander Alpha:** Move left.",
        );

        assert_eq!(response, "Move left.");
    }

    #[test]
    fn plain_text_chat_parser_falls_back_to_raw_text_when_prefix_is_malformed() {
        let response =
            normalize_plain_text_chat_response("agent_alpha", None, "**agent_alpha* Move left.");

        assert_eq!(response, "**agent_alpha* Move left.");
    }

    #[test]
    fn runtime_config_rejects_duplicate_profile_ids() {
        let result = LlmRuntimeConfig {
            profiles: vec![
                LlmRuntimeProfileConfig::default(),
                LlmRuntimeProfileConfig {
                    id: String::from("default"),
                    ..LlmRuntimeProfileConfig::default()
                },
            ],
            routing: LlmTaskRoutingConfig::default(),
        }
        .validate();

        assert!(matches!(
            result,
            Err(super::LlmRuntimeError::DuplicateProfileId(profile_id)) if profile_id == "default"
        ));
    }

    #[test]
    fn runtime_config_rejects_missing_routed_profile_ids() {
        let result = LlmRuntimeConfig {
            profiles: vec![LlmRuntimeProfileConfig::default()],
            routing: LlmTaskRoutingConfig {
                turn_generation_profile: String::from("missing"),
                fact_extraction_profile: String::from("default"),
            },
        }
        .validate();

        assert!(matches!(
            result,
            Err(super::LlmRuntimeError::MissingRoutedProfile { route, profile_id })
                if route == "turn_generation_profile" && profile_id == "missing"
        ));
    }

    #[test]
    fn runtime_config_coerces_zero_worker_count_to_one() {
        let validated = LlmRuntimeConfig {
            profiles: vec![LlmRuntimeProfileConfig {
                worker_count: 0,
                ..LlmRuntimeProfileConfig::default()
            }],
            routing: LlmTaskRoutingConfig::default(),
        }
        .validate()
        .expect("config should validate");

        assert_eq!(validated.profiles[0].worker_count, 1);
    }

    fn demo_world() -> LlmWorldContext {
        LlmWorldContext {
            world_view: json!({
                "agent_name": "Commander Alpha",
                "persona": "Calm field commander who speaks with crisp urgency.",
                "current_goal": "Secure the bridge before the scout advances."
            }),
            world_schema: json!({}),
        }
    }
}
