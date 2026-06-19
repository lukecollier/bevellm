extern crate self as bevellm;

pub mod conversation;
pub mod llm;
pub use bevellm_macros::LLMActions;
pub use conversation::*;
pub use llm::{
    LlmConversationEntry, LlmConversationEntryKind, LlmConversationFact, LlmConversationState,
    LlmModel, LlmRuntime, LlmRuntimeConfig, LlmRuntimeProfileConfig, LlmTaskRoutingConfig,
    LlmToolCall, LlmTurn,
};

use bevy_app::{App, Plugin, PreUpdate, Update};
use bevy_ecs::prelude::*;
use crossbeam_channel as crossbeam;
use log::{debug, info};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Instant;

/// A request from the game to an LLM agent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum LlmResponseMode {
    /// Structured JSON for tool-capable or machine-parsed flows.
    #[default]
    StructuredJson,
    /// Plain text for local world-facing speech.
    PlainTextChat,
}

/// Model-specific tool calling behavior for a runtime profile.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum LlmToolCallingMode {
    /// Do not expose tools to the model.
    #[default]
    Disabled,
    /// Use the model's recommended tool-calling format.
    Auto,
    /// Expose tools and let the model choose when to use them.
    Native,
    /// Expose tools using the agentic XML style used by SmolLM3.
    AgenticXml,
    /// Expose tools using a python-style call format.
    Python,
}

impl LlmToolCallingMode {
    pub fn resolve_for_model(self, model: LlmModel) -> Self {
        match self {
            Self::Auto => match model {
                LlmModel::SmolLM3_3BQ4KM => Self::AgenticXml,
                LlmModel::SmolLM2_360MInstruct
                | LlmModel::SmolLM2_1_7BInstructQ4KM
                | LlmModel::Qwen2_5_1_5BInstructQ2K
                | LlmModel::Qwen2_5_1_5BInstructQ4KM => Self::Native,
            },
            other => other,
        }
    }
}

/// A tool definition exposed to an LLM request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl LlmToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// A request from the game to an LLM agent.
#[derive(Debug, Clone, Event, Serialize, Deserialize, PartialEq)]
pub struct LlmRequest {
    /// Logical agent identifier.
    pub agent: String,
    /// Optional conversation thread identifier. `None` means stateless.
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// Optional request-scoped world context override.
    ///
    /// When present, this replaces the stored per-agent world snapshot for this
    /// request only. This is useful for task-specific reduced views.
    #[serde(default)]
    pub world_override: Option<LlmWorldContext>,
    /// Prompt text to send to the agent.
    pub prompt: String,
    /// Output format expected from the model.
    #[serde(default)]
    pub response_mode: LlmResponseMode,
    /// Tool definitions exposed to the model for this request.
    #[serde(default)]
    pub tools: Vec<LlmToolDefinition>,
}

/// Conversation lifecycle commands for named LLM threads.
#[derive(Debug, Clone, Event, Serialize, Deserialize, PartialEq, Eq)]
pub enum LlmConversationCommand {
    Reset {
        agent: String,
        conversation_id: String,
    },
    Compact {
        agent: String,
        conversation_id: String,
    },
    AppendMessage {
        agent: String,
        conversation_id: String,
        message: String,
    },
}

/// One-shot conversation generation commands.
#[derive(Debug, Clone, Event, Serialize, Deserialize, PartialEq, Eq)]
pub enum LlmConversationGenerationCommand {
    GenerateConversation {
        session_id: String,
        participants: Vec<String>,
        initial_message: String,
        facts: Vec<String>,
    },
    CancelConversation {
        session_id: String,
    },
}

/// Planned conversation utterance for Bevy-side playback.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlannedUtterance {
    pub speaker: String,
    pub text: String,
    #[serde(default)]
    pub tool_calls: Vec<LlmToolCall>,
}

/// Full generated conversation transcript for Bevy-side playback.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeneratedConversation {
    pub session_id: String,
    pub participants: Vec<String>,
    pub utterances: Vec<PlannedUtterance>,
}

/// One-shot conversation generation events emitted by the runtime.
#[derive(Debug, Clone, Event, Serialize, Deserialize, PartialEq)]
pub enum LlmConversationGenerationEvent {
    ConversationGenerated { conversation: GeneratedConversation },
    ConversationGenerationFailed { session_id: String, reason: String },
}

/// Direct fact extraction commands for Bevy-side conversation data.
#[derive(Debug, Clone, Event, Serialize, Deserialize, PartialEq)]
pub enum LlmFactExtractionCommand {
    ExtractFactFromUtterance {
        request_id: String,
        session_id: String,
        utterance_index: usize,
        utterance: PlannedUtterance,
    },
}

/// Prompt-level fact storage commands for Bevy-side inputs.
#[derive(Debug, Clone, Event, Serialize, Deserialize, PartialEq)]
pub enum LlmFactStoreCommand {
    StoreFactsFromPrompt {
        request_id: String,
        text: String,
        agents: Vec<String>,
    },
}

/// Fact extraction results emitted by the runtime.
#[derive(Debug, Clone, Event, Serialize, Deserialize, PartialEq)]
pub enum LlmFactExtractionEvent {
    FactsExtractedFromUtterance {
        request_id: String,
        session_id: String,
        utterance_index: usize,
        utterance: PlannedUtterance,
        facts: Vec<LlmConversationFact>,
    },
    FactExtractionFailedForUtterance {
        request_id: String,
        session_id: String,
        utterance_index: usize,
        utterance: PlannedUtterance,
        reason: String,
    },
}

/// Prompt-level fact storage results emitted by the runtime.
#[derive(Debug, Clone, Event, Serialize, Deserialize, PartialEq)]
pub enum LlmFactStoreEvent {
    FactsStored {
        request_id: String,
        agents: Vec<String>,
        facts: Vec<LlmConversationFact>,
    },
    FactStorageFailed {
        request_id: String,
        agents: Vec<String>,
        reason: String,
    },
}

/// A message returned by the LLM worker.
#[derive(Debug, Clone, Event, Serialize, Deserialize, PartialEq)]
pub struct LlmResponse {
    /// Logical agent identifier.
    pub agent: String,
    /// Optional conversation thread identifier associated with the response.
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// Text returned by the model, if any.
    pub response: Option<String>,
    /// All tool calls captured from the model output.
    #[serde(default)]
    pub tool_calls: Vec<LlmToolCall>,
}

/// Lightweight summary of the tools exposed to a model.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmToolSet {
    names: Vec<String>,
    definitions: Vec<LlmToolDefinition>,
}

impl LlmToolSet {
    pub fn new(names: Vec<String>, definitions: Vec<LlmToolDefinition>) -> Self {
        Self { names, definitions }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.iter().any(|candidate| candidate == name)
    }

    pub fn definitions(&self) -> &[LlmToolDefinition] {
        &self.definitions
    }
}

/// Marks an ECS entity as an LLM-backed agent.
#[derive(Debug, Clone, Component, Serialize, Deserialize, PartialEq, Eq)]
pub struct LLMAgent {
    /// Logical agent identifier used for routing requests and responses.
    pub id: String,
}

/// Serializable world-facing context associated with an LLM agent.
#[derive(Debug, Clone, Component, PartialEq)]
pub struct LLMAgentWorld<T>(pub T);

impl<T> Deref for LLMAgentWorld<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for LLMAgentWorld<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Normalized world payload passed to the worker thread.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmWorldContext {
    pub world_view: Value,
    pub world_schema: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, SystemSet)]
enum LlmBridgeSet {
    TrackAgents,
    SyncWorld,
    ForwardRequests,
    DrainResponses,
}

#[derive(Resource, Default)]
struct LLMAgentRegistry {
    ids: HashMap<Entity, String>,
}

#[derive(Resource, Default)]
struct LlmWorldSyncInstalled;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ConversationKey {
    agent: String,
    conversation_id: String,
}

const GENERATED_CONVERSATION_MAX_UTTERANCES: usize = 6;
const FACT_STORE_CONVERSATION_ID: &str = "fact_store";

#[derive(Debug, Clone)]
enum WorkerRequest {
    Standard(LlmRequest),
    GeneratedConversation(ConversationGenerationRequest),
    ExtractFacts(FactExtractionRequest),
    StoreFacts(FactStoreRequest),
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutedTaskKind {
    TurnGeneration,
    FactExtraction,
    FactPromptExtraction,
}

#[derive(Debug, Clone)]
struct ConversationGenerationRequest {
    session_id: String,
    generation: u64,
    participants: Vec<String>,
    initial_message: String,
    facts: Vec<String>,
    participant_profiles: Vec<ConversationParticipantProfile>,
    tool_calling: LlmToolCallingMode,
}

#[derive(Debug, Clone)]
enum WorkerTurn {
    Standard(LlmTurn),
    GeneratedConversation(ConversationGenerationResult),
    ExtractedFacts(FactExtractionResult),
    StoredFacts(FactStoreResult),
}

#[derive(Debug)]
enum ProfileJob {
    RefreshFacts {
        request: FactRefreshTask,
        response_tx: crossbeam::Sender<Result<LlmConversationState, llm::LlmRuntimeError>>,
    },
    GenerateTurn {
        request: TurnGenerationTask,
        response_tx: crossbeam::Sender<Result<LlmTurn, llm::LlmRuntimeError>>,
    },
    ExtractFactsFromPrompt {
        request: FactStoreRequest,
        response_tx: crossbeam::Sender<Result<Vec<LlmConversationFact>, llm::LlmRuntimeError>>,
    },
}

#[derive(Debug, Clone)]
struct FactRefreshTask {
    conversation: LlmConversationState,
    world: Option<LlmWorldContext>,
    cancel_flag: Option<Arc<AtomicBool>>,
}

#[derive(Debug, Clone)]
struct TurnGenerationTask {
    agent: String,
    conversation_id: Option<String>,
    prompt: String,
    response_mode: LlmResponseMode,
    facts: Vec<LlmConversationFact>,
    tools: Vec<LlmToolDefinition>,
    world: Option<LlmWorldContext>,
    conversation: Option<LlmConversationState>,
    session_participants: Option<Vec<String>>,
    speaker_labels: Option<HashMap<String, String>>,
    cancel_flag: Option<Arc<AtomicBool>>,
}

#[derive(Debug, Clone)]
struct ConversationParticipantProfile {
    id: String,
    display_name: String,
    persona: String,
    current_goal: String,
}

#[derive(Debug, Clone)]
struct ConversationGenerationResult {
    session_id: String,
    generation: u64,
    result: Result<Vec<PlannedUtterance>, String>,
}

#[derive(Debug, Clone)]
struct FactExtractionRequest {
    request_id: String,
    session_id: String,
    utterance_index: usize,
    utterance: PlannedUtterance,
}

#[derive(Debug, Clone)]
struct FactStoreRequest {
    request_id: String,
    text: String,
    agents: Vec<String>,
}

#[derive(Debug, Clone)]
struct FactExtractionResult {
    request_id: String,
    session_id: String,
    utterance_index: usize,
    utterance: PlannedUtterance,
    result: Result<Vec<LlmConversationFact>, String>,
}

#[derive(Debug, Clone)]
struct FactStoreResult {
    request_id: String,
    agents: Vec<String>,
    result: Result<Vec<LlmConversationFact>, String>,
}

#[derive(Debug, Clone, Default)]
struct ConversationGenerationState {
    participants: Vec<String>,
    generation: u64,
}

trait RuntimeExecutor: Send {
    fn refresh_conversation_facts(
        &mut self,
        conversation: &mut LlmConversationState,
        world: Option<&LlmWorldContext>,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<(), llm::LlmRuntimeError>;

    fn generate_turn(
        &mut self,
        request: &TurnGenerationTask,
    ) -> Result<LlmTurn, llm::LlmRuntimeError>;

    fn extract_facts_from_prompt(
        &mut self,
        prompt: &str,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<Vec<LlmConversationFact>, llm::LlmRuntimeError>;
}

struct MistralRuntimeExecutor {
    runtime: LlmRuntime,
}

impl RuntimeExecutor for MistralRuntimeExecutor {
    fn refresh_conversation_facts(
        &mut self,
        conversation: &mut LlmConversationState,
        world: Option<&LlmWorldContext>,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<(), llm::LlmRuntimeError> {
        self.runtime
            .refresh_conversation_facts(conversation, world, cancel_flag)
    }

    fn generate_turn(
        &mut self,
        request: &TurnGenerationTask,
    ) -> Result<LlmTurn, llm::LlmRuntimeError> {
        self.runtime.generate_turn(
            &request.agent,
            request.conversation_id.as_deref(),
            &request.prompt,
            request.response_mode,
            request.cancel_flag.as_deref(),
            &request.facts,
            &request.tools,
            request.world.as_ref(),
            request.conversation.as_ref(),
            request.session_participants.as_deref(),
            request.speaker_labels.as_ref(),
        )
    }

    fn extract_facts_from_prompt(
        &mut self,
        prompt: &str,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<Vec<LlmConversationFact>, llm::LlmRuntimeError> {
        self.runtime.extract_facts_from_prompt(prompt, cancel_flag)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum BridgeOutput {
    Response(LlmTurn),
    ConversationEvent(LlmConversationGenerationEvent),
    FactEvent(LlmFactExtractionEvent),
    FactStoreEvent(LlmFactStoreEvent),
}

/// Error returned when an LLM action cannot be handed off to the game thread.
#[derive(Debug, thiserror::Error)]
pub enum LLMActionError {
    #[error("failed to dispatch LLM action: {0}")]
    Dispatch(String),
}

/// In-process inbox for actions emitted by model tool calls.
///
/// The tools send actions from whatever thread they run on. A Bevy system
/// drains this inbox and converts the actions into Bevy events.
#[derive(Resource, Clone)]
pub struct LLMActionInbox<E> {
    receiver: Arc<Mutex<mpsc::Receiver<E>>>,
}

impl<E> LLMActionInbox<E> {
    /// Create an inbox from a standard library channel receiver.
    pub fn new(receiver: mpsc::Receiver<E>) -> Self {
        Self {
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }

    /// Drain all pending actions into a vector.
    pub fn drain(&self) -> Vec<E> {
        let mut drained = Vec::new();

        if let Ok(receiver) = self.receiver.lock() {
            while let Ok(action) = receiver.try_recv() {
                drained.push(action);
            }
        }

        drained
    }
}

/// Create a sender/inbox pair for LLM actions.
pub fn llm_action_channel<E>() -> (mpsc::Sender<E>, LLMActionInbox<E>) {
    let (sender, receiver) = mpsc::channel();
    (sender, LLMActionInbox::new(receiver))
}

/// In-process inbox for prompts emitted by Bevy systems.
///
/// Game systems send requests from whatever thread they run on. A Bevy system
/// drains this inbox and converts the requests into Bevy events.
#[derive(Resource, Clone)]
pub struct LlmRequestInbox {
    receiver: Arc<Mutex<mpsc::Receiver<LlmRequest>>>,
}

impl LlmRequestInbox {
    /// Create an inbox from a standard library channel receiver.
    pub fn new(receiver: mpsc::Receiver<LlmRequest>) -> Self {
        Self {
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }

    /// Drain all pending requests into a vector.
    pub fn drain(&self) -> Vec<LlmRequest> {
        let mut drained = Vec::new();

        if let Ok(receiver) = self.receiver.lock() {
            while let Ok(request) = receiver.try_recv() {
                drained.push(request);
            }
        }

        drained
    }
}

/// Create a sender/inbox pair for LLM requests.
pub fn llm_request_channel() -> (mpsc::Sender<LlmRequest>, LlmRequestInbox) {
    let (sender, receiver) = mpsc::channel();
    (sender, LlmRequestInbox::new(receiver))
}

/// Add the inbox and drain system for LLM requests.
pub fn install_llm_requests(app: &mut App, inbox: LlmRequestInbox) {
    app.add_event::<LlmRequest>()
        .insert_resource(inbox)
        .add_systems(PreUpdate, drain_llm_requests);
}

/// Add the inbox and drain system for LLM conversation commands.
pub fn install_llm_conversation_commands(
    app: &mut App,
    inbox: LLMActionInbox<LlmConversationCommand>,
) {
    app.add_event::<LlmConversationCommand>()
        .insert_resource(inbox)
        .add_systems(PreUpdate, drain_llm_actions::<LlmConversationCommand>);
}

/// Add the inbox and drain system for one-shot LLM conversation generation commands.
pub fn install_llm_conversation_generation_commands(
    app: &mut App,
    inbox: LLMActionInbox<LlmConversationGenerationCommand>,
) {
    app.add_event::<LlmConversationGenerationCommand>()
        .insert_resource(inbox)
        .add_systems(
            PreUpdate,
            drain_llm_actions::<LlmConversationGenerationCommand>,
        );
}

/// Add the inbox and drain system for direct fact extraction commands.
pub fn install_llm_fact_extraction_commands(
    app: &mut App,
    inbox: LLMActionInbox<LlmFactExtractionCommand>,
) {
    app.add_event::<LlmFactExtractionCommand>()
        .insert_resource(inbox)
        .add_systems(PreUpdate, drain_llm_actions::<LlmFactExtractionCommand>);
}

/// Add the inbox and drain system for prompt-level fact storage commands.
pub fn install_llm_fact_store_commands(app: &mut App, inbox: LLMActionInbox<LlmFactStoreCommand>) {
    app.add_event::<LlmFactStoreCommand>()
        .insert_resource(inbox)
        .add_systems(PreUpdate, drain_llm_actions::<LlmFactStoreCommand>);
}

/// Drain pending requests from the inbox and publish them as Bevy events.
pub fn drain_llm_requests(inbox: Res<LlmRequestInbox>, mut writer: EventWriter<LlmRequest>) {
    for request in inbox.drain() {
        debug!(
            "llm request drained from inbox: agent={} conversation={:?} mode={:?} has_world_override={} prompt={}",
            request.agent,
            request.conversation_id,
            request.response_mode,
            request.world_override.is_some(),
            request.prompt
        );
        writer.send(request);
    }
}

/// A shared bridge between Bevy and a pooled LLM runtime.
#[derive(Resource, Clone)]
pub struct LlmRuntimeBridge {
    request_tx: crossbeam::Sender<WorkerRequest>,
    turns: crossbeam::Receiver<WorkerTurn>,
    worlds: Arc<Mutex<HashMap<String, LlmWorldContext>>>,
    conversations: Arc<Mutex<HashMap<String, HashMap<String, llm::LlmConversationState>>>>,
    active_conversations: Arc<Mutex<HashSet<ConversationKey>>>,
    pending_conversation_requests: Arc<Mutex<HashMap<ConversationKey, VecDeque<LlmRequest>>>>,
    active_cancellations: Arc<Mutex<HashMap<ConversationKey, Arc<AtomicBool>>>>,
    generated_conversations: Arc<Mutex<HashMap<String, ConversationGenerationState>>>,
    generated_conversation_cancellations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    turn_generation_tool_calling: LlmToolCallingMode,
}

impl LlmRuntimeBridge {
    fn new(
        request_tx: crossbeam::Sender<WorkerRequest>,
        turns: crossbeam::Receiver<WorkerTurn>,
        worlds: Arc<Mutex<HashMap<String, LlmWorldContext>>>,
        conversations: Arc<Mutex<HashMap<String, HashMap<String, llm::LlmConversationState>>>>,
        active_conversations: Arc<Mutex<HashSet<ConversationKey>>>,
        pending_conversation_requests: Arc<Mutex<HashMap<ConversationKey, VecDeque<LlmRequest>>>>,
        active_cancellations: Arc<Mutex<HashMap<ConversationKey, Arc<AtomicBool>>>>,
        generated_conversations: Arc<Mutex<HashMap<String, ConversationGenerationState>>>,
        generated_conversation_cancellations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
        turn_generation_tool_calling: LlmToolCallingMode,
    ) -> Self {
        Self {
            request_tx,
            turns,
            worlds,
            conversations,
            active_conversations,
            pending_conversation_requests,
            active_cancellations,
            generated_conversations,
            generated_conversation_cancellations,
            turn_generation_tool_calling,
        }
    }

    /// Forward a request to the background worker.
    pub fn send(&self, request: LlmRequest) -> Result<(), LLMActionError> {
        debug!(
            "llm request forwarded to worker: agent={} conversation={:?} mode={:?} has_world_override={} prompt={}",
            request.agent,
            request.conversation_id,
            request.response_mode,
            request.world_override.is_some(),
            request.prompt
        );
        if let Some(conversation_id) = &request.conversation_id {
            let key = ConversationKey {
                agent: request.agent.clone(),
                conversation_id: conversation_id.clone(),
            };
            let mut active = self
                .active_conversations
                .lock()
                .map_err(|err| LLMActionError::Dispatch(err.to_string()))?;
            if active.insert(key.clone()) {
                drop(active);
                self.request_tx
                    .send(WorkerRequest::Standard(request))
                    .map_err(|err| LLMActionError::Dispatch(err.to_string()))
            } else {
                drop(active);
                if let Ok(cancellations) = self.active_cancellations.lock() {
                    if let Some(cancel_flag) = cancellations.get(&key) {
                        cancel_flag.store(true, Ordering::Relaxed);
                    }
                }
                self.pending_conversation_requests
                    .lock()
                    .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
                    .entry(key)
                    .and_modify(|queue| {
                        queue.clear();
                        queue.push_back(request.clone());
                    })
                    .or_insert_with(|| VecDeque::from([request]));
                Ok(())
            }
        } else {
            self.request_tx
                .send(WorkerRequest::Standard(request))
                .map_err(|err| LLMActionError::Dispatch(err.to_string()))
        }
    }

    /// Replace the worker-side world snapshot for an agent.
    pub fn set_world(&self, agent: String, world: LlmWorldContext) -> Result<(), LLMActionError> {
        debug!("llm world snapshot forwarded to worker: agent={agent}");
        self.worlds
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
            .insert(agent, world);
        Ok(())
    }

    /// Clear the worker-side world snapshot for an agent.
    pub fn clear_world(&self, agent: String) -> Result<(), LLMActionError> {
        debug!("llm world snapshot cleared in worker: agent={agent}");
        self.worlds
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
            .remove(&agent);
        self.conversations
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
            .remove(&agent);
        self.active_conversations
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
            .retain(|key| key.agent != agent);
        self.active_cancellations
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
            .retain(|key, _| key.agent != agent);
        self.pending_conversation_requests
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
            .retain(|key, _| key.agent != agent);
        Ok(())
    }

    /// Reset a named conversation thread for an agent.
    pub fn reset_conversation(
        &self,
        agent: String,
        conversation_id: String,
    ) -> Result<(), LLMActionError> {
        let key = ConversationKey {
            agent: agent.clone(),
            conversation_id: conversation_id.clone(),
        };
        let mut conversations = self
            .conversations
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?;
        if let Some(agent_conversations) = conversations.get_mut(&agent) {
            agent_conversations.remove(&conversation_id);
            if agent_conversations.is_empty() {
                conversations.remove(&agent);
            }
        }
        self.pending_conversation_requests
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
            .remove(&key);
        self.active_cancellations
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
            .remove(&key);
        Ok(())
    }

    /// Compact a named conversation thread for an agent.
    pub fn compact_conversation(
        &self,
        agent: String,
        conversation_id: String,
    ) -> Result<(), LLMActionError> {
        let mut conversations = self
            .conversations
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?;
        if let Some(agent_conversations) = conversations.get_mut(&agent) {
            if let Some(conversation) = agent_conversations.get_mut(&conversation_id) {
                llm::compact_conversation_state(conversation);
            }
        }
        Ok(())
    }

    /// Append an external message to a named conversation thread for an agent.
    pub fn append_conversation_message(
        &self,
        agent: String,
        conversation_id: String,
        message: String,
    ) -> Result<(), LLMActionError> {
        let mut conversations = self
            .conversations
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?;
        let agent_conversations = conversations.entry(agent).or_default();
        let conversation = agent_conversations
            .entry(conversation_id)
            .or_insert_with(llm::LlmConversationState::default);
        llm::append_conversation_control(conversation, message);
        Ok(())
    }

    /// Snapshot a named conversation thread for an agent.
    pub fn conversation_state(
        &self,
        agent: &str,
        conversation_id: &str,
    ) -> Result<Option<LlmConversationState>, LLMActionError> {
        let conversations = self
            .conversations
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?;
        Ok(conversations
            .get(agent)
            .and_then(|threads| threads.get(conversation_id))
            .cloned())
    }

    fn drain_outputs(&self) -> Vec<BridgeOutput> {
        let mut drained = Vec::new();
        while let Ok(turn) = self.turns.try_recv() {
            match turn {
                WorkerTurn::Standard(turn) => {
                    let _ = self.release_next_conversation_request(&turn);
                    drained.push(BridgeOutput::Response(turn));
                }
                WorkerTurn::GeneratedConversation(turn) => {
                    drained.extend(self.handle_generated_conversation(turn))
                }
                WorkerTurn::ExtractedFacts(turn) => drained.push(self.handle_extracted_facts(turn)),
                WorkerTurn::StoredFacts(turn) => drained.push(self.handle_stored_facts(turn)),
            }
        }

        drained
    }

    fn handle_generation_command(
        &self,
        command: LlmConversationGenerationCommand,
    ) -> Result<Vec<LlmConversationGenerationEvent>, LLMActionError> {
        match command {
            LlmConversationGenerationCommand::GenerateConversation {
                session_id,
                participants,
                initial_message,
                facts,
            } => self.generate_conversation(session_id, participants, initial_message, facts),
            LlmConversationGenerationCommand::CancelConversation { session_id } => {
                self.cancel_generated_conversation(&session_id)?;
                Ok(Vec::new())
            }
        }
    }

    fn generate_conversation(
        &self,
        session_id: String,
        participants: Vec<String>,
        initial_message: String,
        facts: Vec<String>,
    ) -> Result<Vec<LlmConversationGenerationEvent>, LLMActionError> {
        let participants = dedupe_participants(participants);
        if participants.is_empty() {
            return Ok(vec![
                LlmConversationGenerationEvent::ConversationGenerationFailed {
                    session_id,
                    reason: String::from("conversation requires at least one participant"),
                },
            ]);
        }

        self.cancel_generated_conversation(&session_id)?;
        self.generated_conversation_cancellations
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
            .insert(session_id.clone(), Arc::new(AtomicBool::new(false)));
        let request = {
            let worlds = self
                .worlds
                .lock()
                .map_err(|err| LLMActionError::Dispatch(err.to_string()))?;
            let mut conversations = self
                .generated_conversations
                .lock()
                .map_err(|err| LLMActionError::Dispatch(err.to_string()))?;
            let generation = conversations
                .get(&session_id)
                .map(|state| state.generation)
                .unwrap_or(0)
                .wrapping_add(1);
            conversations.insert(
                session_id.clone(),
                ConversationGenerationState {
                    participants: participants.clone(),
                    generation,
                },
            );
            let participant_profiles = participants
                .iter()
                .map(|participant| ConversationParticipantProfile {
                    id: participant.clone(),
                    display_name: world_display_name(worlds.get(participant), participant),
                    persona: world_string_field(worlds.get(participant), "persona"),
                    current_goal: world_string_field(worlds.get(participant), "current_goal"),
                })
                .collect();
            ConversationGenerationRequest {
                session_id: session_id.clone(),
                generation,
                participants,
                initial_message,
                facts,
                participant_profiles,
                tool_calling: self.turn_generation_tool_calling,
            }
        };

        self.request_tx
            .send(WorkerRequest::GeneratedConversation(request))
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?;
        Ok(Vec::new())
    }

    fn handle_generated_conversation(
        &self,
        turn: ConversationGenerationResult,
    ) -> Vec<BridgeOutput> {
        let state = match self.generated_conversations.lock() {
            Ok(conversations) => conversations.get(&turn.session_id).cloned(),
            Err(err) => {
                eprintln!("failed to lock generated conversations: {err}");
                None
            }
        };
        let Some(state) = state else {
            return Vec::new();
        };
        if state.generation != turn.generation {
            return Vec::new();
        }

        if let Ok(mut cancellations) = self.generated_conversation_cancellations.lock() {
            cancellations.remove(&turn.session_id);
        }

        match turn.result {
            Ok(utterances) => vec![BridgeOutput::ConversationEvent(
                LlmConversationGenerationEvent::ConversationGenerated {
                    conversation: GeneratedConversation {
                        session_id: turn.session_id,
                        participants: state.participants,
                        utterances,
                    },
                },
            )],
            Err(reason) => vec![BridgeOutput::ConversationEvent(
                LlmConversationGenerationEvent::ConversationGenerationFailed {
                    session_id: turn.session_id,
                    reason,
                },
            )],
        }
    }

    fn handle_fact_extraction_command(
        &self,
        command: LlmFactExtractionCommand,
    ) -> Result<(), LLMActionError> {
        match command {
            LlmFactExtractionCommand::ExtractFactFromUtterance {
                request_id,
                session_id,
                utterance_index,
                utterance,
            } => self
                .request_tx
                .send(WorkerRequest::ExtractFacts(FactExtractionRequest {
                    request_id,
                    session_id,
                    utterance_index,
                    utterance,
                }))
                .map_err(|err| LLMActionError::Dispatch(err.to_string())),
        }
    }

    fn handle_fact_store_command(
        &self,
        command: LlmFactStoreCommand,
    ) -> Result<(), LLMActionError> {
        match command {
            LlmFactStoreCommand::StoreFactsFromPrompt {
                request_id,
                text,
                agents,
            } => self
                .request_tx
                .send(WorkerRequest::StoreFacts(FactStoreRequest {
                    request_id,
                    text,
                    agents,
                }))
                .map_err(|err| LLMActionError::Dispatch(err.to_string())),
        }
    }

    fn handle_extracted_facts(&self, turn: FactExtractionResult) -> BridgeOutput {
        match turn.result {
            Ok(facts) => {
                BridgeOutput::FactEvent(LlmFactExtractionEvent::FactsExtractedFromUtterance {
                    request_id: turn.request_id,
                    session_id: turn.session_id,
                    utterance_index: turn.utterance_index,
                    utterance: turn.utterance,
                    facts,
                })
            }
            Err(reason) => {
                BridgeOutput::FactEvent(LlmFactExtractionEvent::FactExtractionFailedForUtterance {
                    request_id: turn.request_id,
                    session_id: turn.session_id,
                    utterance_index: turn.utterance_index,
                    utterance: turn.utterance,
                    reason,
                })
            }
        }
    }

    fn handle_stored_facts(&self, turn: FactStoreResult) -> BridgeOutput {
        match turn.result {
            Ok(facts) => BridgeOutput::FactStoreEvent(LlmFactStoreEvent::FactsStored {
                request_id: turn.request_id,
                agents: turn.agents,
                facts,
            }),
            Err(reason) => BridgeOutput::FactStoreEvent(LlmFactStoreEvent::FactStorageFailed {
                request_id: turn.request_id,
                agents: turn.agents,
                reason,
            }),
        }
    }

    fn cancel_generated_conversation(&self, session_id: &str) -> Result<(), LLMActionError> {
        if let Some(flag) = self
            .generated_conversation_cancellations
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
            .get(session_id)
            .cloned()
        {
            flag.store(true, Ordering::Relaxed);
        }
        Ok(())
    }

    fn release_next_conversation_request(&self, turn: &LlmTurn) -> Result<(), LLMActionError> {
        let Some(conversation_id) = &turn.conversation_id else {
            return Ok(());
        };
        let key = ConversationKey {
            agent: turn.agent.clone(),
            conversation_id: conversation_id.clone(),
        };
        release_next_conversation_request_internal(
            &self.request_tx,
            &self.active_conversations,
            &self.pending_conversation_requests,
            &self.active_cancellations,
            &key,
        )
    }
}

fn release_next_conversation_request_internal(
    request_tx: &crossbeam::Sender<WorkerRequest>,
    active_conversations: &Arc<Mutex<HashSet<ConversationKey>>>,
    pending_conversation_requests: &Arc<Mutex<HashMap<ConversationKey, VecDeque<LlmRequest>>>>,
    active_cancellations: &Arc<Mutex<HashMap<ConversationKey, Arc<AtomicBool>>>>,
    key: &ConversationKey,
) -> Result<(), LLMActionError> {
    active_cancellations
        .lock()
        .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
        .remove(key);

    let next_request = {
        let mut pending = pending_conversation_requests
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?;
        let next_request = pending.get_mut(key).and_then(VecDeque::pop_front);
        if pending.get(key).is_some_and(VecDeque::is_empty) {
            pending.remove(key);
        }
        next_request
    };

    if let Some(next_request) = next_request {
        request_tx
            .send(WorkerRequest::Standard(next_request))
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?;
    } else {
        active_conversations
            .lock()
            .map_err(|err| LLMActionError::Dispatch(err.to_string()))?
            .remove(key);
    }

    Ok(())
}

fn dispatch_profile_job<T>(
    profile_senders: &BTreeMap<String, crossbeam::Sender<ProfileJob>>,
    profile_id: &str,
    build_job: impl FnOnce(crossbeam::Sender<T>) -> ProfileJob,
) -> Result<T, String>
where
    T: Send + 'static,
{
    let (response_tx, response_rx) = crossbeam::bounded(1);
    let Some(profile_tx) = profile_senders.get(profile_id) else {
        return Err(format!("missing runtime profile `{profile_id}`"));
    };
    profile_tx
        .send(build_job(response_tx))
        .map_err(|err| err.to_string())?;
    response_rx.recv().map_err(|err| err.to_string())
}

fn collect_agent_facts(
    conversations: &Arc<Mutex<HashMap<String, HashMap<String, llm::LlmConversationState>>>>,
    agent: &str,
) -> Vec<LlmConversationFact> {
    conversations
        .lock()
        .ok()
        .and_then(|conversations| conversations.get(agent).cloned())
        .map(|agent_conversations| {
            let mut merged = Vec::new();
            for conversation in agent_conversations.values() {
                llm::merge_conversation_facts(&mut merged, conversation.facts.clone());
            }
            merged
        })
        .unwrap_or_default()
}

fn is_llm_error_response(response: &str) -> bool {
    response.starts_with("LLM error:")
}

fn world_display_name(world: Option<&LlmWorldContext>, fallback: &str) -> String {
    world_string_value(world, "agent_name").unwrap_or_else(|| fallback.to_string())
}

fn world_string_field(world: Option<&LlmWorldContext>, field: &str) -> String {
    world_string_value(world, field).unwrap_or_default()
}

fn world_string_value(world: Option<&LlmWorldContext>, field: &str) -> Option<String> {
    world.and_then(|world| {
        world
            .world_view
            .get(field)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

fn process_standard_request(
    request: LlmRequest,
    routing: &LlmTaskRoutingConfig,
    profile_senders: &BTreeMap<String, crossbeam::Sender<ProfileJob>>,
    turn_tx: &crossbeam::Sender<WorkerTurn>,
    request_tx: &crossbeam::Sender<WorkerRequest>,
    worlds: &Arc<Mutex<HashMap<String, LlmWorldContext>>>,
    conversations: &Arc<Mutex<HashMap<String, HashMap<String, llm::LlmConversationState>>>>,
    active_conversations: &Arc<Mutex<HashSet<ConversationKey>>>,
    pending_conversation_requests: &Arc<Mutex<HashMap<ConversationKey, VecDeque<LlmRequest>>>>,
    active_cancellations: &Arc<Mutex<HashMap<ConversationKey, Arc<AtomicBool>>>>,
) {
    let request_started = Instant::now();
    let conversation_key =
        request
            .conversation_id
            .as_ref()
            .map(|conversation_id| ConversationKey {
                agent: request.agent.clone(),
                conversation_id: conversation_id.clone(),
            });
    let cancel_flag = conversation_key.as_ref().and_then(|key| {
        active_cancellations.lock().ok().map(|mut cancellations| {
            cancellations
                .entry(key.clone())
                .or_insert_with(|| Arc::new(AtomicBool::new(false)))
                .clone()
        })
    });
    let world = worlds.lock().ok().and_then(|worlds| {
        request
            .world_override
            .clone()
            .or_else(|| worlds.get(&request.agent).cloned())
    });
    let mut conversation = request
        .conversation_id
        .as_ref()
        .and_then(|conversation_id| {
            conversations.lock().ok().and_then(|mut conversations| {
                let agent_conversations = conversations.entry(request.agent.clone()).or_default();
                for conversation in agent_conversations.values_mut() {
                    llm::expire_conversation_facts(conversation);
                }
                Some(
                    agent_conversations
                        .entry(conversation_id.clone())
                        .or_insert_with(llm::LlmConversationState::default)
                        .clone(),
                )
            })
        });

    if let Some(snapshot) = conversation.clone() {
        match dispatch_profile_job(
            profile_senders,
            &routing.fact_extraction_profile,
            |response_tx| ProfileJob::RefreshFacts {
                request: FactRefreshTask {
                    conversation: snapshot,
                    world: world.clone(),
                    cancel_flag: cancel_flag.clone(),
                },
                response_tx,
            },
        ) {
            Ok(Ok(refreshed)) => {
                if let Some(conversation_id) = &request.conversation_id {
                    if let Ok(mut conversations) = conversations.lock() {
                        let agent_conversations =
                            conversations.entry(request.agent.clone()).or_default();
                        agent_conversations.insert(conversation_id.clone(), refreshed.clone());
                    }
                }
                conversation = Some(refreshed);
            }
            Ok(Err(llm::LlmRuntimeError::Cancelled)) => {
                if let Some(key) = &conversation_key {
                    let _ = release_next_conversation_request_internal(
                        request_tx,
                        active_conversations,
                        pending_conversation_requests,
                        active_cancellations,
                        key,
                    );
                }
                return;
            }
            Ok(Err(err)) => {
                eprintln!(
                    "failed to refresh conversation facts for agent={} conversation={:?}: {}",
                    request.agent, request.conversation_id, err
                );
            }
            Err(err) => {
                eprintln!(
                    "failed to dispatch conversation fact refresh for agent={} conversation={:?}: {}",
                    request.agent, request.conversation_id, err
                );
            }
        }
    }

    let agent_facts = collect_agent_facts(conversations, &request.agent);
    debug!(
        "llm coordinator request start: agent={} conversation={:?} mode={:?} turn_profile={} fact_profile={} has_world={} has_world_override={} has_conversation={} fact_count={} prompt={}",
        request.agent,
        request.conversation_id,
        request.response_mode,
        routing.turn_generation_profile,
        routing.fact_extraction_profile,
        world.is_some(),
        request.world_override.is_some(),
        conversation.is_some(),
        agent_facts.len(),
        request.prompt
    );

    if cancel_flag
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
    {
        if let Some(key) = &conversation_key {
            let _ = release_next_conversation_request_internal(
                request_tx,
                active_conversations,
                pending_conversation_requests,
                active_cancellations,
                key,
            );
        }
        return;
    }

    let turn_request = TurnGenerationTask {
        agent: request.agent.clone(),
        conversation_id: request.conversation_id.clone(),
        prompt: request.prompt.clone(),
        response_mode: request.response_mode,
        facts: agent_facts,
        tools: request.tools.clone(),
        world,
        conversation: conversation.clone(),
        session_participants: None,
        speaker_labels: None,
        cancel_flag: cancel_flag.clone(),
    };

    let turn = match dispatch_profile_job(
        profile_senders,
        &routing.turn_generation_profile,
        |response_tx| ProfileJob::GenerateTurn {
            request: turn_request,
            response_tx,
        },
    ) {
        Ok(Ok(turn)) => turn,
        Ok(Err(llm::LlmRuntimeError::Cancelled)) => {
            debug!(
                "llm coordinator request cancelled after {:?}: agent={} conversation={:?}",
                request_started.elapsed(),
                request.agent,
                request.conversation_id
            );
            if let Some(key) = &conversation_key {
                let _ = release_next_conversation_request_internal(
                    request_tx,
                    active_conversations,
                    pending_conversation_requests,
                    active_cancellations,
                    key,
                );
            }
            return;
        }
        Ok(Err(err)) => LlmTurn {
            agent: request.agent.clone(),
            conversation_id: request.conversation_id.clone(),
            response: format!("LLM error: {err}"),
            tool_calls: Vec::new(),
        },
        Err(err) => LlmTurn {
            agent: request.agent.clone(),
            conversation_id: request.conversation_id.clone(),
            response: format!("LLM error: {err}"),
            tool_calls: Vec::new(),
        },
    };

    if let Some(conversation_id) = &request.conversation_id {
        if is_llm_error_response(&turn.response) {
            debug!(
                "llm coordinator skipped persisting runtime error turn: agent={} conversation={:?} response={}",
                request.agent, request.conversation_id, turn.response
            );
        } else if let Ok(mut conversations) = conversations.lock() {
            let agent_conversations = conversations.entry(request.agent.clone()).or_default();
            let conversation_state = agent_conversations
                .entry(conversation_id.clone())
                .or_insert_with(llm::LlmConversationState::default);
            llm::append_conversation_turn(
                conversation_state,
                &request.agent,
                &request.prompt,
                &turn.response,
                &turn.tool_calls,
            );
        }
    }

    debug!(
        "llm coordinator request done in {:?}: agent={} conversation={:?} response={} tool_calls={}",
        request_started.elapsed(),
        turn.agent,
        turn.conversation_id,
        turn.response,
        turn.tool_calls.len()
    );
    let _ = turn_tx.send(WorkerTurn::Standard(turn));
}

fn process_generated_conversation_request(
    request: ConversationGenerationRequest,
    routing: &LlmTaskRoutingConfig,
    profile_senders: &BTreeMap<String, crossbeam::Sender<ProfileJob>>,
    turn_tx: &crossbeam::Sender<WorkerTurn>,
    generated_conversation_cancellations: &Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
) {
    let cancel_flag = generated_conversation_cancellations
        .lock()
        .ok()
        .map(|mut cancellations| {
            cancellations
                .entry(request.session_id.clone())
                .or_insert_with(|| Arc::new(AtomicBool::new(false)))
                .clone()
        });

    let turn_request = TurnGenerationTask {
        agent: request
            .participants
            .first()
            .cloned()
            .unwrap_or_else(|| String::from("conversation_generator")),
        conversation_id: Some(request.session_id.clone()),
        prompt: build_generated_conversation_prompt(&request),
        response_mode: LlmResponseMode::StructuredJson,
        facts: Vec::new(),
        tools: Vec::new(),
        world: None,
        conversation: None,
        session_participants: None,
        speaker_labels: None,
        cancel_flag: cancel_flag.clone(),
    };

    let result = match dispatch_profile_job(
        profile_senders,
        &routing.turn_generation_profile,
        |response_tx| ProfileJob::GenerateTurn {
            request: turn_request,
            response_tx,
        },
    ) {
        Ok(Ok(turn)) => parse_generated_conversation_response(
            &turn.response,
            &request.participants,
            GENERATED_CONVERSATION_MAX_UTTERANCES,
        ),
        Ok(Err(llm::LlmRuntimeError::Cancelled)) => return,
        Ok(Err(err)) => Err(format!("LLM error: {err}")),
        Err(err) => Err(format!("LLM error: {err}")),
    };

    let _ = turn_tx.send(WorkerTurn::GeneratedConversation(
        ConversationGenerationResult {
            session_id: request.session_id,
            generation: request.generation,
            result,
        },
    ));
}

fn process_fact_extraction_request(
    request: FactExtractionRequest,
    routing: &LlmTaskRoutingConfig,
    profile_senders: &BTreeMap<String, crossbeam::Sender<ProfileJob>>,
    turn_tx: &crossbeam::Sender<WorkerTurn>,
) {
    let mut conversation = llm::LlmConversationState::default();
    llm::append_conversation_chat(
        &mut conversation,
        request.utterance.speaker.clone(),
        request.utterance.text.clone(),
        true,
    );

    let result = match dispatch_profile_job(
        profile_senders,
        &routing.fact_extraction_profile,
        |response_tx| ProfileJob::RefreshFacts {
            request: FactRefreshTask {
                conversation,
                world: None,
                cancel_flag: None,
            },
            response_tx,
        },
    ) {
        Ok(Ok(conversation)) => Ok(conversation.facts),
        Ok(Err(llm::LlmRuntimeError::Cancelled)) => Err(String::from("fact extraction cancelled")),
        Ok(Err(err)) => Err(format!("LLM error: {err}")),
        Err(err) => Err(format!("LLM error: {err}")),
    };

    let _ = turn_tx.send(WorkerTurn::ExtractedFacts(FactExtractionResult {
        request_id: request.request_id,
        session_id: request.session_id,
        utterance_index: request.utterance_index,
        utterance: request.utterance,
        result,
    }));
}

fn process_fact_store_request(
    request: FactStoreRequest,
    routing: &LlmTaskRoutingConfig,
    profile_senders: &BTreeMap<String, crossbeam::Sender<ProfileJob>>,
    turn_tx: &crossbeam::Sender<WorkerTurn>,
    conversations: &Arc<Mutex<HashMap<String, HashMap<String, llm::LlmConversationState>>>>,
) {
    let agents = dedupe_participants(request.agents);
    if agents.is_empty() {
        let _ = turn_tx.send(WorkerTurn::StoredFacts(FactStoreResult {
            request_id: request.request_id,
            agents,
            result: Err(String::from("at least one agent is required")),
        }));
        return;
    }

    let result = match dispatch_profile_job(
        profile_senders,
        &routing.fact_extraction_profile,
        |response_tx| ProfileJob::ExtractFactsFromPrompt {
            request: FactStoreRequest {
                request_id: request.request_id.clone(),
                text: request.text.clone(),
                agents: agents.clone(),
            },
            response_tx,
        },
    ) {
        Ok(Ok(facts)) => {
            if let Ok(mut conversations) = conversations.lock() {
                for agent in &agents {
                    let agent_conversations = conversations.entry(agent.clone()).or_default();
                    let fact_store = agent_conversations
                        .entry(String::from(FACT_STORE_CONVERSATION_ID))
                        .or_insert_with(llm::LlmConversationState::default);
                    llm::merge_conversation_facts(&mut fact_store.facts, facts.clone());
                }
            }
            Ok(facts)
        }
        Ok(Err(llm::LlmRuntimeError::Cancelled)) => Err(String::from("fact storage cancelled")),
        Ok(Err(err)) => Err(format!("LLM error: {err}")),
        Err(err) => Err(format!("LLM error: {err}")),
    };

    let _ = turn_tx.send(WorkerTurn::StoredFacts(FactStoreResult {
        request_id: request.request_id,
        agents,
        result,
    }));
}

fn dedupe_participants(participants: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    for participant in participants {
        if seen.insert(participant.clone()) {
            ordered.push(participant);
        }
    }
    ordered
}

fn build_generated_conversation_prompt(request: &ConversationGenerationRequest) -> String {
    let mut content = String::from(
        "Generate a complete short in-world conversation for the provided participants.\n\
Write the whole exchange, not a single reply.\n\
Use the participant information below to shape what each person says.\n\
Make the conversation feel like these specific participants talking to each other about the situation.\n\
Return exactly one JSON object.\n\
No extra text.\n\
The JSON object must contain only one field: `utterances`.\n\
`utterances` must be an array of objects.\n\
Each utterance object must contain a `speaker` string field and a `text` string field.\n\
It may also contain a `tool_calls` array for proposed actions.\n\
Each `speaker` must exactly match one of the provided participant ids.\n\
Each `text` must be spoken dialogue only, with no speaker prefix.\n\
If an utterance combines spoken dialogue and an action, keep the dialogue in `text` and put the action in `tool_calls`.\n\
Only leave `text` empty when the utterance is purely an action beat with no spoken line.\n\
Do not include markdown.\n\
Do not include narration or stage directions.\n",
    );
    content.push_str(&format!(
        "Active tool-calling format for proposed actions: {}.\n",
        tool_calling_mode_display(request.tool_calling)
    ));
    content.push_str(&format!(
        "Produce at most {} utterances.\n\n",
        GENERATED_CONVERSATION_MAX_UTTERANCES
    ));
    content.push_str(
        "Conversation requirements:\n\
- The output should be a full conversation arc for this moment.\n\
- Ground the dialogue in the initial situation.\n\
- Let the participants' personas and current goals influence their wording and priorities.\n\
- Prefer including every participant at least once when that fits within the utterance cap.\n\
- Keep each line as spoken dialogue when speech is present, and pair that dialogue with `tool_calls` when the speaker is also acting.\n\
- Do not narrate actions in `text`; use `tool_calls` for proposed actions, but keep the speaker's line in `text` when there is one.\n\n",
    );
    content.push_str("Initial situation:\n");
    content.push_str(&request.initial_message);
    if !request.facts.is_empty() {
        content.push_str("\n\nConversation facts gathered from prior rounds:\n");
        for fact in &request.facts {
            let fact = fact.trim();
            if !fact.is_empty() {
                content.push_str(&format!("- {fact}\n"));
            }
        }
    }
    content.push_str("\n\nParticipant information:\n");
    for participant in &request.participant_profiles {
        content.push_str(&format!(
            "- display_name: {}\n  persona: {}\n  current_goal: {}\n",
            participant.display_name,
            empty_to_placeholder(&participant.persona),
            empty_to_placeholder(&participant.current_goal),
        ));
    }
    content.push_str("\nValid speaker ids for `speaker` fields:\n");
    for participant in &request.participant_profiles {
        content.push_str(&format!("- {}\n", participant.id));
    }
    content.push_str(
        "\nOutput example:\n{\"utterances\":[{\"speaker\":\"agent_alpha\",\"text\":\"Move to the bridge, but stay low.\",\"tool_calls\":[{\"tool\":\"secure_bridge_approach\",\"arguments\":{\"approach\":\"east\"}}]},{\"speaker\":\"agent_bravo\",\"text\":\"I’m warning you, the eastern path is exposed.\",\"tool_calls\":[{\"tool\":\"warn_about_exposed_path\",\"arguments\":{\"path\":\"eastern\",\"severity\":\"high\"}}]}]}\n",
    );
    content
}

fn tool_calling_mode_display(mode: LlmToolCallingMode) -> &'static str {
    match mode {
        LlmToolCallingMode::Disabled => "disabled",
        LlmToolCallingMode::Auto => "auto",
        LlmToolCallingMode::Native => "native",
        LlmToolCallingMode::AgenticXml => "agentic XML",
        LlmToolCallingMode::Python => "python",
    }
}

fn empty_to_placeholder(value: &str) -> &str {
    if value.trim().is_empty() {
        "unspecified"
    } else {
        value
    }
}

#[derive(Debug, Deserialize)]
struct GeneratedConversationJson {
    utterances: Vec<PlannedUtteranceJson>,
}

#[derive(Debug, Deserialize)]
struct PlannedUtteranceJson {
    speaker: String,
    text: String,
    #[serde(default)]
    tool_calls: Vec<LlmToolCall>,
}

fn parse_generated_conversation_response(
    response: &str,
    participants: &[String],
    max_utterances: usize,
) -> Result<Vec<PlannedUtterance>, String> {
    let parsed: GeneratedConversationJson = serde_json::from_str(response.trim())
        .map_err(|err| format!("invalid conversation JSON: {err}"))?;
    if parsed.utterances.len() > max_utterances {
        return Err(format!(
            "conversation exceeded max utterance count of {max_utterances}"
        ));
    }

    let allowed = participants.iter().cloned().collect::<HashSet<_>>();
    let mut utterances = Vec::with_capacity(parsed.utterances.len());
    for utterance in parsed.utterances {
        if !allowed.contains(&utterance.speaker) {
            return Err(format!("unknown speaker id `{}`", utterance.speaker));
        }
        let text = utterance.text.trim();
        if text.is_empty() && utterance.tool_calls.is_empty() {
            return Err(format!(
                "blank utterance from speaker `{}`",
                utterance.speaker
            ));
        }
        utterances.push(PlannedUtterance {
            speaker: utterance.speaker,
            text: text.to_string(),
            tool_calls: utterance.tool_calls,
        });
    }
    Ok(utterances)
}

/// Bevy plugin that installs the runtime bridge with a configurable worker pool.
#[derive(Debug, Clone)]
pub struct LlmRuntimePlugin {
    pub config: LlmRuntimeConfig,
}

impl Default for LlmRuntimePlugin {
    fn default() -> Self {
        Self {
            config: LlmRuntimeConfig::default(),
        }
    }
}

impl Plugin for LlmRuntimePlugin {
    fn build(&self, app: &mut App) {
        let bridge = spawn_llm_runtime_bridge(self.config.clone())
            .expect("failed to start LLM runtime bridge");
        install_llm_runtime_bridge(app, bridge);
    }
}

/// Create the LLM worker pool and return the bridge resource.
pub fn spawn_llm_runtime_bridge(
    config: LlmRuntimeConfig,
) -> Result<LlmRuntimeBridge, llm::LlmRuntimeError> {
    let factory: Arc<
        dyn Fn(
                &LlmRuntimeProfileConfig,
                usize,
            ) -> Result<Box<dyn RuntimeExecutor>, llm::LlmRuntimeError>
            + Send
            + Sync,
    > = Arc::new(|profile, worker_index| {
        let runtime = LlmRuntime::load(profile)?.with_worker_index(worker_index);
        Ok(Box::new(MistralRuntimeExecutor { runtime }))
    });
    spawn_llm_runtime_bridge_with_factory(config, factory)
}

fn spawn_llm_runtime_bridge_with_factory(
    config: LlmRuntimeConfig,
    factory: Arc<
        dyn Fn(
                &LlmRuntimeProfileConfig,
                usize,
            ) -> Result<Box<dyn RuntimeExecutor>, llm::LlmRuntimeError>
            + Send
            + Sync,
    >,
) -> Result<LlmRuntimeBridge, llm::LlmRuntimeError> {
    let validated = config.validate()?;
    info!(
        "starting llm runtime profiles={:?} routing=turn:{} facts:{}",
        validated
            .profiles
            .iter()
            .map(|profile| format!("{}:{} workers", profile.id, profile.worker_count))
            .collect::<Vec<_>>(),
        validated.routing.turn_generation_profile,
        validated.routing.fact_extraction_profile
    );

    let (request_tx, request_rx) = crossbeam::unbounded::<WorkerRequest>();
    let (turn_tx, turn_rx) = crossbeam::unbounded::<WorkerTurn>();
    let worlds = Arc::new(Mutex::new(HashMap::<String, LlmWorldContext>::new()));
    let conversations = Arc::new(Mutex::new(HashMap::<
        String,
        HashMap<String, llm::LlmConversationState>,
    >::new()));
    let active_conversations = Arc::new(Mutex::new(HashSet::<ConversationKey>::new()));
    let pending_conversation_requests = Arc::new(Mutex::new(HashMap::<
        ConversationKey,
        VecDeque<LlmRequest>,
    >::new()));
    let active_cancellations = Arc::new(Mutex::new(
        HashMap::<ConversationKey, Arc<AtomicBool>>::new(),
    ));
    let generated_conversations = Arc::new(Mutex::new(HashMap::<
        String,
        ConversationGenerationState,
    >::new()));
    let generated_conversation_cancellations =
        Arc::new(Mutex::new(HashMap::<String, Arc<AtomicBool>>::new()));
    let mut profile_senders = BTreeMap::<String, crossbeam::Sender<ProfileJob>>::new();
    for profile in &validated.profiles {
        let (profile_tx, profile_rx) = crossbeam::unbounded::<ProfileJob>();
        profile_senders.insert(profile.id.clone(), profile_tx);

        for worker_index in 0..profile.worker_count {
            let profile_rx = profile_rx.clone();
            let profile = profile.clone();
            let factory = Arc::clone(&factory);
            thread::spawn(move || {
                let mut executor = match factory(&profile, worker_index) {
                    Ok(executor) => executor,
                    Err(err) => {
                        eprintln!(
                            "failed to start llm worker profile={} worker={}: {}",
                            profile.id, worker_index, err
                        );
                        return;
                    }
                };
                info!(
                    "llm worker started profile={} worker={}",
                    profile.id, worker_index
                );

                while let Ok(job) = profile_rx.recv() {
                    match job {
                        ProfileJob::RefreshFacts {
                            mut request,
                            response_tx,
                        } => {
                            let result = executor
                                .refresh_conversation_facts(
                                    &mut request.conversation,
                                    request.world.as_ref(),
                                    request.cancel_flag.as_deref(),
                                )
                                .map(|_| request.conversation);
                            let _ = response_tx.send(result);
                        }
                        ProfileJob::GenerateTurn {
                            request,
                            response_tx,
                        } => {
                            let _ = response_tx.send(executor.generate_turn(&request));
                        }
                        ProfileJob::ExtractFactsFromPrompt {
                            request,
                            response_tx,
                        } => {
                            let _ = response_tx
                                .send(executor.extract_facts_from_prompt(&request.text, None));
                        }
                    }
                }
            });
        }
    }

    let turn_generation_tool_calling = validated
        .profiles
        .iter()
        .find(|profile| profile.id == validated.routing.turn_generation_profile)
        .map(|profile| profile.tool_calling.resolve_for_model(profile.model))
        .ok_or_else(|| llm::LlmRuntimeError::MissingRoutedProfile {
            route: "turn_generation_profile",
            profile_id: validated.routing.turn_generation_profile.clone(),
        })?;

    let routing = validated.routing.clone();
    let coordinator_turn_tx = turn_tx.clone();
    let coordinator_worlds = Arc::clone(&worlds);
    let coordinator_conversations = Arc::clone(&conversations);
    let coordinator_active_conversations = Arc::clone(&active_conversations);
    let coordinator_pending_conversation_requests = Arc::clone(&pending_conversation_requests);
    let coordinator_active_cancellations = Arc::clone(&active_cancellations);
    let coordinator_generated_conversation_cancellations =
        Arc::clone(&generated_conversation_cancellations);
    let coordinator_request_tx = request_tx.clone();
    thread::spawn(move || {
        while let Ok(request) = request_rx.recv() {
            match request {
                WorkerRequest::Standard(request) => {
                    let profile_senders = profile_senders.clone();
                    let routing = routing.clone();
                    let turn_tx = coordinator_turn_tx.clone();
                    let worlds = Arc::clone(&coordinator_worlds);
                    let conversations = Arc::clone(&coordinator_conversations);
                    let active_conversations = Arc::clone(&coordinator_active_conversations);
                    let pending_conversation_requests =
                        Arc::clone(&coordinator_pending_conversation_requests);
                    let active_cancellations = Arc::clone(&coordinator_active_cancellations);
                    let request_tx = coordinator_request_tx.clone();
                    thread::spawn(move || {
                        process_standard_request(
                            request,
                            &routing,
                            &profile_senders,
                            &turn_tx,
                            &request_tx,
                            &worlds,
                            &conversations,
                            &active_conversations,
                            &pending_conversation_requests,
                            &active_cancellations,
                        );
                    });
                }
                WorkerRequest::GeneratedConversation(request) => {
                    let profile_senders = profile_senders.clone();
                    let routing = routing.clone();
                    let turn_tx = coordinator_turn_tx.clone();
                    let generated_conversation_cancellations =
                        Arc::clone(&coordinator_generated_conversation_cancellations);
                    thread::spawn(move || {
                        process_generated_conversation_request(
                            request,
                            &routing,
                            &profile_senders,
                            &turn_tx,
                            &generated_conversation_cancellations,
                        );
                    });
                }
                WorkerRequest::ExtractFacts(request) => {
                    let profile_senders = profile_senders.clone();
                    let routing = routing.clone();
                    let turn_tx = coordinator_turn_tx.clone();
                    thread::spawn(move || {
                        process_fact_extraction_request(
                            request,
                            &routing,
                            &profile_senders,
                            &turn_tx,
                        );
                    });
                }
                WorkerRequest::StoreFacts(request) => {
                    let profile_senders = profile_senders.clone();
                    let routing = routing.clone();
                    let turn_tx = coordinator_turn_tx.clone();
                    let conversations = Arc::clone(&coordinator_conversations);
                    thread::spawn(move || {
                        process_fact_store_request(
                            request,
                            &routing,
                            &profile_senders,
                            &turn_tx,
                            &conversations,
                        );
                    });
                }
            }
        }
    });

    Ok(LlmRuntimeBridge::new(
        request_tx,
        turn_rx,
        worlds,
        conversations,
        active_conversations,
        pending_conversation_requests,
        active_cancellations,
        generated_conversations,
        generated_conversation_cancellations,
        turn_generation_tool_calling,
    ))
}

/// Install the bridge and wire Bevy events to the background worker.
pub fn install_llm_runtime_bridge(app: &mut App, bridge: LlmRuntimeBridge) {
    app.configure_sets(
        PreUpdate,
        (
            LlmBridgeSet::TrackAgents,
            LlmBridgeSet::SyncWorld,
            LlmBridgeSet::ForwardRequests,
            LlmBridgeSet::DrainResponses,
        )
            .chain(),
    )
    .add_event::<LlmRequest>()
    .add_event::<LlmConversationCommand>()
    .add_event::<LlmConversationGenerationCommand>()
    .add_event::<LlmConversationGenerationEvent>()
    .add_event::<LlmFactExtractionCommand>()
    .add_event::<LlmFactExtractionEvent>()
    .add_event::<LlmFactStoreCommand>()
    .add_event::<LlmFactStoreEvent>()
    .add_event::<LlmResponse>()
    .insert_resource(LLMAgentRegistry::default())
    .insert_resource(bridge)
    .add_systems(
        PreUpdate,
        track_llm_agents.in_set(LlmBridgeSet::TrackAgents),
    )
    .add_systems(
        PreUpdate,
        cleanup_removed_agents.in_set(LlmBridgeSet::SyncWorld),
    )
    .add_systems(
        PreUpdate,
        (
            forward_llm_conversation_commands,
            forward_llm_conversation_generation_commands,
            forward_llm_fact_extraction_commands,
            forward_llm_fact_store_commands,
            forward_llm_requests,
        )
            .chain()
            .in_set(LlmBridgeSet::ForwardRequests),
    )
    .add_systems(
        PreUpdate,
        drain_llm_turns.in_set(LlmBridgeSet::DrainResponses),
    );
}

fn forward_llm_requests(bridge: Res<LlmRuntimeBridge>, mut requests: EventReader<LlmRequest>) {
    for request in requests.read() {
        debug!(
            "llm request read from bevy event: agent={} conversation={:?} mode={:?} prompt={}",
            request.agent, request.conversation_id, request.response_mode, request.prompt
        );
        if let Err(err) = bridge.send(request.clone()) {
            eprintln!("failed to forward llm request: {err}");
        }
    }
}

fn forward_llm_conversation_commands(
    bridge: Res<LlmRuntimeBridge>,
    mut commands: EventReader<LlmConversationCommand>,
) {
    for command in commands.read() {
        let result = match command {
            LlmConversationCommand::Reset {
                agent,
                conversation_id,
            } => bridge.reset_conversation(agent.clone(), conversation_id.clone()),
            LlmConversationCommand::Compact {
                agent,
                conversation_id,
            } => bridge.compact_conversation(agent.clone(), conversation_id.clone()),
            LlmConversationCommand::AppendMessage {
                agent,
                conversation_id,
                message,
            } => bridge.append_conversation_message(
                agent.clone(),
                conversation_id.clone(),
                message.clone(),
            ),
        };

        if let Err(err) = result {
            eprintln!("failed to forward llm conversation command: {err}");
        }
    }
}

fn forward_llm_conversation_generation_commands(
    bridge: Res<LlmRuntimeBridge>,
    mut commands: EventReader<LlmConversationGenerationCommand>,
    mut events: EventWriter<LlmConversationGenerationEvent>,
) {
    for command in commands.read() {
        match bridge.handle_generation_command(command.clone()) {
            Ok(emitted) => {
                for event in emitted {
                    events.send(event);
                }
            }
            Err(err) => eprintln!("failed to forward llm conversation generation command: {err}"),
        }
    }
}

fn forward_llm_fact_extraction_commands(
    bridge: Res<LlmRuntimeBridge>,
    mut commands: EventReader<LlmFactExtractionCommand>,
) {
    for command in commands.read() {
        if let Err(err) = bridge.handle_fact_extraction_command(command.clone()) {
            eprintln!("failed to forward llm fact extraction command: {err}");
        }
    }
}

fn forward_llm_fact_store_commands(
    bridge: Res<LlmRuntimeBridge>,
    mut commands: EventReader<LlmFactStoreCommand>,
) {
    for command in commands.read() {
        if let Err(err) = bridge.handle_fact_store_command(command.clone()) {
            eprintln!("failed to forward llm fact store command: {err}");
        }
    }
}

fn drain_llm_turns(
    bridge: Res<LlmRuntimeBridge>,
    mut responses: EventWriter<LlmResponse>,
    mut conversation_events: EventWriter<LlmConversationGenerationEvent>,
    mut fact_events: EventWriter<LlmFactExtractionEvent>,
    mut fact_store_events: EventWriter<LlmFactStoreEvent>,
) {
    for output in bridge.drain_outputs() {
        match output {
            BridgeOutput::Response(turn) => {
                debug!(
                    "llm turn drained into bevy events: agent={} response={} tool_calls={}",
                    turn.agent,
                    turn.response,
                    turn.tool_calls.len()
                );
                let response = if turn.tool_calls.is_empty() {
                    if turn.response.is_empty() {
                        None
                    } else {
                        Some(turn.response)
                    }
                } else {
                    None
                };
                responses.send(LlmResponse {
                    agent: turn.agent,
                    conversation_id: turn.conversation_id,
                    response,
                    tool_calls: turn.tool_calls,
                });
            }
            BridgeOutput::ConversationEvent(event) => {
                conversation_events.send(event);
            }
            BridgeOutput::FactEvent(event) => {
                fact_events.send(event);
            }
            BridgeOutput::FactStoreEvent(event) => {
                fact_store_events.send(event);
            }
        }
    }
}

fn track_llm_agents(
    agents: Query<(Entity, &LLMAgent), Or<(Added<LLMAgent>, Changed<LLMAgent>)>>,
    mut registry: ResMut<LLMAgentRegistry>,
) {
    for (entity, agent) in agents.iter() {
        registry.ids.insert(entity, agent.id.clone());
    }
}

fn cleanup_removed_agents(
    mut removed_agents: RemovedComponents<LLMAgent>,
    mut registry: ResMut<LLMAgentRegistry>,
    bridge: Res<LlmRuntimeBridge>,
) {
    for entity in removed_agents.read() {
        let Some(agent_id) = registry.ids.remove(&entity) else {
            continue;
        };
        if let Err(err) = bridge.clear_world(agent_id) {
            eprintln!("failed to clear llm world for removed agent: {err}");
        }
    }
}

/// Install world synchronization systems for a concrete world-view type.
pub fn install_llm_world_sync<T>(app: &mut App)
where
    T: Serialize + JsonSchema + Clone + Send + Sync + 'static,
{
    if !app.world().contains_resource::<LlmWorldSyncInstalled>() {
        app.insert_resource(LlmWorldSyncInstalled);
    }

    app.add_systems(
        PreUpdate,
        (sync_llm_agent_world::<T>, cleanup_removed_agent_world::<T>)
            .in_set(LlmBridgeSet::SyncWorld),
    );
}

/// Build a normalized worker-facing world payload from a typed request view.
pub fn llm_world_context<T>(world: &T) -> Result<LlmWorldContext, serde_json::Error>
where
    T: Serialize + JsonSchema,
{
    serialize_world_context(world)
}

fn sync_llm_agent_world<T>(
    bridge: Res<LlmRuntimeBridge>,
    worlds: Query<
        (&LLMAgent, &LLMAgentWorld<T>),
        Or<(Added<LLMAgentWorld<T>>, Changed<LLMAgentWorld<T>>)>,
    >,
) where
    T: Serialize + JsonSchema + Clone + Send + Sync + 'static,
{
    for (agent, world) in worlds.iter() {
        let payload = match serialize_world_context(&world.0) {
            Ok(payload) => payload,
            Err(err) => {
                eprintln!(
                    "failed to serialize llm world for agent={}: {err}",
                    agent.id
                );
                continue;
            }
        };

        if let Err(err) = bridge.set_world(agent.id.clone(), payload) {
            eprintln!("failed to forward llm world snapshot: {err}");
        }
    }
}

fn cleanup_removed_agent_world<T>(
    mut removed_worlds: RemovedComponents<LLMAgentWorld<T>>,
    agents: Query<&LLMAgent>,
    bridge: Res<LlmRuntimeBridge>,
) where
    T: Serialize + JsonSchema + Clone + Send + Sync + 'static,
{
    for entity in removed_worlds.read() {
        let Ok(agent) = agents.get(entity) else {
            continue;
        };
        if let Err(err) = bridge.clear_world(agent.id.clone()) {
            eprintln!("failed to clear llm world for removed world component: {err}");
        }
    }
}

fn serialize_world_context<T>(world: &T) -> Result<LlmWorldContext, serde_json::Error>
where
    T: Serialize + JsonSchema,
{
    Ok(LlmWorldContext {
        world_view: serde_json::to_value(world)?,
        world_schema: serde_json::to_value(schema_for!(T))?,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        BridgeOutput, FACT_STORE_CONVERSATION_ID, LlmConversationGenerationCommand,
        LlmConversationGenerationEvent, LlmFactExtractionCommand, LlmFactExtractionEvent,
        LlmFactStoreCommand, LlmFactStoreEvent, LlmRequest, LlmResponseMode, LlmRuntimeBridge,
        LlmRuntimeProfileConfig, LlmTaskRoutingConfig, LlmToolCall, LlmToolCallingMode,
        LlmWorldContext, PlannedUtterance, RoutedTaskKind, RuntimeExecutor, TurnGenerationTask,
        build_generated_conversation_prompt, llm, parse_generated_conversation_response,
        spawn_llm_runtime_bridge_with_factory,
    };
    use crate::LlmModel;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TaskRecord {
        profile_id: String,
        task: RoutedTaskKind,
    }

    struct StubRuntimeExecutor {
        profile_id: String,
        records: Arc<Mutex<Vec<TaskRecord>>>,
        response_delay: Duration,
    }

    impl RuntimeExecutor for StubRuntimeExecutor {
        fn refresh_conversation_facts(
            &mut self,
            conversation: &mut llm::LlmConversationState,
            _world: Option<&LlmWorldContext>,
            cancel_flag: Option<&AtomicBool>,
        ) -> Result<(), llm::LlmRuntimeError> {
            self.records.lock().expect("records lock").push(TaskRecord {
                profile_id: self.profile_id.clone(),
                task: RoutedTaskKind::FactExtraction,
            });
            if cancel_flag.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return Err(llm::LlmRuntimeError::Cancelled);
            }
            conversation.facts.push(llm::LlmConversationFact {
                fact: format!("fact-from-{}", self.profile_id),
                importance: 0.8,
                expiry: String::from("unknown"),
            });
            conversation.processed_entry_count = conversation.entries.len();
            Ok(())
        }

        fn generate_turn(
            &mut self,
            request: &TurnGenerationTask,
        ) -> Result<llm::LlmTurn, llm::LlmRuntimeError> {
            self.records.lock().expect("records lock").push(TaskRecord {
                profile_id: self.profile_id.clone(),
                task: RoutedTaskKind::TurnGeneration,
            });
            let start = Instant::now();
            while start.elapsed() < self.response_delay {
                if request
                    .cancel_flag
                    .as_ref()
                    .is_some_and(|flag| flag.load(Ordering::Relaxed))
                {
                    return Err(llm::LlmRuntimeError::Cancelled);
                }
                thread::sleep(Duration::from_millis(10));
            }

            Ok(llm::LlmTurn {
                agent: request.agent.clone(),
                conversation_id: request.conversation_id.clone(),
                response: if request
                    .prompt
                    .contains("The JSON object must contain only one field: `utterances`.")
                {
                    String::from(
                        "{\"utterances\":[{\"speaker\":\"alpha\",\"text\":\"Hold the bridge.\"},{\"speaker\":\"bravo\",\"text\":\"I see movement on the east path.\"}]}",
                    )
                } else {
                    format!("turn-from-{}: {}", self.profile_id, request.prompt)
                },
                tool_calls: Vec::new(),
            })
        }

        fn extract_facts_from_prompt(
            &mut self,
            prompt: &str,
            _cancel_flag: Option<&AtomicBool>,
        ) -> Result<Vec<llm::LlmConversationFact>, llm::LlmRuntimeError> {
            self.records.lock().expect("records lock").push(TaskRecord {
                profile_id: self.profile_id.clone(),
                task: RoutedTaskKind::FactPromptExtraction,
            });
            Ok(vec![llm::LlmConversationFact {
                fact: format!("fact-from-prompt: {}", prompt),
                importance: 0.8,
                expiry: String::from("unknown"),
            }])
        }
    }

    fn stub_bridge(response_delay: Duration) -> (LlmRuntimeBridge, Arc<Mutex<Vec<TaskRecord>>>) {
        let records = Arc::new(Mutex::new(Vec::<TaskRecord>::new()));
        let records_for_factory = Arc::clone(&records);
        let factory = Arc::new(
            move |profile: &LlmRuntimeProfileConfig, _worker_index: usize| {
                Ok(Box::new(StubRuntimeExecutor {
                    profile_id: profile.id.clone(),
                    records: Arc::clone(&records_for_factory),
                    response_delay,
                }) as Box<dyn RuntimeExecutor>)
            },
        );

        let bridge = spawn_llm_runtime_bridge_with_factory(
            llm::LlmRuntimeConfig {
                profiles: vec![
                    LlmRuntimeProfileConfig {
                        id: String::from("qwen"),
                        worker_count: 1,
                        ..LlmRuntimeProfileConfig::default()
                    },
                    LlmRuntimeProfileConfig {
                        id: String::from("smollm2"),
                        worker_count: 1,
                        ..LlmRuntimeProfileConfig::default()
                    },
                ],
                routing: LlmTaskRoutingConfig {
                    turn_generation_profile: String::from("qwen"),
                    fact_extraction_profile: String::from("smollm2"),
                },
            },
            factory,
        )
        .expect("stub bridge should spawn");
        (bridge, records)
    }

    fn wait_for_outputs(
        bridge: &LlmRuntimeBridge,
        expected: usize,
        timeout: Duration,
    ) -> Vec<BridgeOutput> {
        let started = Instant::now();
        loop {
            let outputs = bridge.drain_outputs();
            if outputs.len() >= expected {
                return outputs;
            }
            assert!(started.elapsed() < timeout, "timed out waiting for outputs");
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn generated_conversation_json_parses_valid_utterances() {
        let utterances = parse_generated_conversation_response(
            "{\"utterances\":[{\"speaker\":\"alpha\",\"text\":\"Hold the bridge.\"},{\"speaker\":\"bravo\",\"text\":\"I see movement.\"}]}",
            &[String::from("alpha"), String::from("bravo")],
            6,
        )
        .expect("conversation should parse");
        assert_eq!(
            utterances,
            vec![
                PlannedUtterance {
                    speaker: String::from("alpha"),
                    text: String::from("Hold the bridge."),
                    tool_calls: Vec::new(),
                },
                PlannedUtterance {
                    speaker: String::from("bravo"),
                    text: String::from("I see movement."),
                    tool_calls: Vec::new(),
                },
            ]
        );
    }

    #[test]
    fn generated_conversation_rejects_unknown_speaker() {
        let err = parse_generated_conversation_response(
            "{\"utterances\":[{\"speaker\":\"charlie\",\"text\":\"Hold the bridge.\"}]}",
            &[String::from("alpha"), String::from("bravo")],
            6,
        )
        .expect_err("speaker should be rejected");
        assert!(err.contains("unknown speaker id"));
    }

    #[test]
    fn generated_conversation_rejects_blank_utterance_text() {
        let err = parse_generated_conversation_response(
            "{\"utterances\":[{\"speaker\":\"alpha\",\"text\":\"   \"}]}",
            &[String::from("alpha")],
            6,
        )
        .expect_err("blank text should be rejected");
        assert!(err.contains("blank utterance"));
    }

    #[test]
    fn generated_conversation_rejects_oversized_transcript() {
        let err = parse_generated_conversation_response(
            "{\"utterances\":[{\"speaker\":\"alpha\",\"text\":\"1\"},{\"speaker\":\"alpha\",\"text\":\"2\"},{\"speaker\":\"alpha\",\"text\":\"3\"},{\"speaker\":\"alpha\",\"text\":\"4\"},{\"speaker\":\"alpha\",\"text\":\"5\"},{\"speaker\":\"alpha\",\"text\":\"6\"},{\"speaker\":\"alpha\",\"text\":\"7\"}]}",
            &[String::from("alpha")],
            6,
        )
        .expect_err("oversized transcript should be rejected");
        assert!(err.contains("max utterance count"));
    }

    #[test]
    fn standard_request_uses_turn_generation_profile() {
        let (bridge, records) = stub_bridge(Duration::from_millis(0));

        bridge
            .send(LlmRequest {
                agent: String::from("alpha"),
                conversation_id: None,
                world_override: None,
                prompt: String::from("Plan the move."),
                response_mode: LlmResponseMode::StructuredJson,
                tools: Vec::new(),
            })
            .expect("request should dispatch");

        let outputs = wait_for_outputs(&bridge, 1, Duration::from_secs(1));
        assert!(matches!(outputs.as_slice(), [BridgeOutput::Response(_)]));
        assert_eq!(
            records.lock().expect("records lock").as_slice(),
            &[TaskRecord {
                profile_id: String::from("qwen"),
                task: RoutedTaskKind::TurnGeneration,
            }]
        );
    }

    #[test]
    fn generated_conversation_uses_turn_generation_profile_only() {
        let (bridge, records) = stub_bridge(Duration::from_millis(0));

        bridge
            .handle_generation_command(LlmConversationGenerationCommand::GenerateConversation {
                session_id: String::from("session-1"),
                participants: vec![String::from("alpha"), String::from("bravo")],
                initial_message: String::from("Coordinate briefly before moving."),
                facts: vec![String::from("The eastern path is exposed.")],
            })
            .expect("conversation should dispatch");

        let outputs = wait_for_outputs(&bridge, 1, Duration::from_secs(1));
        assert!(matches!(
            outputs.as_slice(),
            [BridgeOutput::ConversationEvent(
                LlmConversationGenerationEvent::ConversationGenerated { .. }
            )]
        ));
        assert_eq!(
            records.lock().expect("records lock").as_slice(),
            &[TaskRecord {
                profile_id: String::from("qwen"),
                task: RoutedTaskKind::TurnGeneration,
            }]
        );
    }

    #[test]
    fn generated_conversation_prompt_includes_supplied_facts() {
        let prompt = build_generated_conversation_prompt(&super::ConversationGenerationRequest {
            session_id: String::from("session-1"),
            generation: 1,
            participants: vec![String::from("alpha"), String::from("bravo")],
            initial_message: String::from("Coordinate briefly before moving."),
            facts: vec![
                String::from("The eastern path is exposed."),
                String::from("The bridge must be secured first."),
            ],
            participant_profiles: vec![
                super::ConversationParticipantProfile {
                    id: String::from("alpha"),
                    display_name: String::from("Commander Alpha"),
                    persona: String::from("Measured squad leader."),
                    current_goal: String::from("Secure the bridge."),
                },
                super::ConversationParticipantProfile {
                    id: String::from("bravo"),
                    display_name: String::from("Scout Bravo"),
                    persona: String::from("Alert reconnaissance scout."),
                    current_goal: String::from("Warn about the east path."),
                },
            ],
            tool_calling: LlmToolCallingMode::Native,
        });

        assert!(prompt.contains("Conversation facts gathered from prior rounds:"));
        assert!(prompt.contains("The eastern path is exposed."));
        assert!(prompt.contains("The bridge must be secured first."));
        assert!(prompt.contains("Active tool-calling format for proposed actions: native."));
        assert!(prompt.contains("keep the dialogue in `text` and put the action in `tool_calls`"));
        assert!(
            prompt.contains("Only leave `text` empty when the utterance is purely an action beat")
        );
        assert!(prompt.contains("tool_calls"));
    }

    #[test]
    fn tool_calling_mode_auto_resolves_by_model() {
        assert_eq!(
            LlmToolCallingMode::Auto.resolve_for_model(LlmModel::Qwen2_5_1_5BInstructQ4KM),
            LlmToolCallingMode::Native
        );
        assert_eq!(
            LlmToolCallingMode::Auto.resolve_for_model(LlmModel::SmolLM3_3BQ4KM),
            LlmToolCallingMode::AgenticXml
        );
    }

    #[test]
    fn generated_conversation_json_parses_tool_only_utterances() {
        let utterances = parse_generated_conversation_response(
            "{\"utterances\":[{\"speaker\":\"alpha\",\"text\":\"\",\"tool_calls\":[{\"tool\":\"advance_to_bridge\",\"arguments\":{\"direction\":\"east\"}}]}]}",
            &[String::from("alpha")],
            6,
        )
        .expect("conversation should parse");
        assert!(utterances[0].text.is_empty());
        assert_eq!(
            utterances[0].tool_calls,
            vec![LlmToolCall {
                tool: String::from("advance_to_bridge"),
                arguments: serde_json::json!({"direction":"east"}),
            }]
        );
    }

    #[test]
    fn fact_refresh_uses_fact_extraction_profile_and_persists_facts() {
        let (bridge, records) = stub_bridge(Duration::from_millis(0));

        {
            let mut conversations = bridge.conversations.lock().expect("conversation lock");
            let state = conversations
                .entry(String::from("alpha"))
                .or_default()
                .entry(String::from("thread"))
                .or_insert_with(llm::LlmConversationState::default);
            llm::append_conversation_chat(state, "bravo", "Supplies are in the shed.", true);
        }

        bridge
            .send(LlmRequest {
                agent: String::from("alpha"),
                conversation_id: Some(String::from("thread")),
                world_override: None,
                prompt: String::from("What matters?"),
                response_mode: LlmResponseMode::StructuredJson,
                tools: Vec::new(),
            })
            .expect("request should dispatch");

        let outputs = wait_for_outputs(&bridge, 1, Duration::from_secs(1));
        assert!(matches!(outputs.as_slice(), [BridgeOutput::Response(_)]));

        let records = records.lock().expect("records lock").clone();
        assert_eq!(
            records,
            vec![
                TaskRecord {
                    profile_id: String::from("smollm2"),
                    task: RoutedTaskKind::FactExtraction,
                },
                TaskRecord {
                    profile_id: String::from("qwen"),
                    task: RoutedTaskKind::TurnGeneration,
                },
            ]
        );

        let conversation = bridge
            .conversation_state("alpha", "thread")
            .expect("conversation state should load")
            .expect("conversation should exist");
        assert!(
            conversation
                .facts
                .iter()
                .any(|fact| fact.fact == "fact-from-smollm2")
        );
    }

    #[test]
    fn direct_fact_extraction_uses_fact_profile_and_emits_event() {
        let (bridge, records) = stub_bridge(Duration::from_millis(0));

        bridge
            .handle_fact_extraction_command(LlmFactExtractionCommand::ExtractFactFromUtterance {
                request_id: String::from("facts-1"),
                session_id: String::from("session-1"),
                utterance_index: 0,
                utterance: PlannedUtterance {
                    speaker: String::from("alpha"),
                    text: String::from("Supplies are in the shed."),
                    tool_calls: Vec::new(),
                },
            })
            .expect("fact extraction should dispatch");

        let outputs = wait_for_outputs(&bridge, 1, Duration::from_secs(1));
        assert!(matches!(
            outputs.as_slice(),
            [BridgeOutput::FactEvent(
                LlmFactExtractionEvent::FactsExtractedFromUtterance { .. }
            )]
        ));
        assert_eq!(
            records.lock().expect("records lock").as_slice(),
            &[TaskRecord {
                profile_id: String::from("smollm2"),
                task: RoutedTaskKind::FactExtraction,
            }]
        );
    }

    #[test]
    fn prompt_fact_storage_uses_fact_profile_and_persists_to_agents() {
        let (bridge, records) = stub_bridge(Duration::from_millis(0));

        bridge
            .handle_fact_store_command(LlmFactStoreCommand::StoreFactsFromPrompt {
                request_id: String::from("store-1"),
                text: String::from("**Alpha:** Alpha ordered the bridge held."),
                agents: vec![String::from("alpha"), String::from("bravo")],
            })
            .expect("fact storage should dispatch");

        let outputs = wait_for_outputs(&bridge, 1, Duration::from_secs(1));
        assert!(matches!(
            outputs.as_slice(),
            [BridgeOutput::FactStoreEvent(
                LlmFactStoreEvent::FactsStored { .. }
            )]
        ));
        assert_eq!(
            records.lock().expect("records lock").as_slice(),
            &[TaskRecord {
                profile_id: String::from("smollm2"),
                task: RoutedTaskKind::FactPromptExtraction,
            }]
        );

        for agent in ["alpha", "bravo"] {
            let conversation = bridge
                .conversation_state(agent, FACT_STORE_CONVERSATION_ID)
                .expect("fact store state should load")
                .expect("fact store should exist");
            assert!(
                conversation
                    .facts
                    .iter()
                    .any(|fact| fact.fact.contains("fact-from-prompt"))
            );
        }
    }

    #[test]
    fn newer_conversation_request_cancels_older_request_and_preserves_ordering() {
        let (bridge, _records) = stub_bridge(Duration::from_millis(150));

        {
            let mut conversations = bridge.conversations.lock().expect("conversation lock");
            conversations
                .entry(String::from("alpha"))
                .or_default()
                .insert(String::from("thread"), llm::LlmConversationState::default());
        }

        bridge
            .send(LlmRequest {
                agent: String::from("alpha"),
                conversation_id: Some(String::from("thread")),
                world_override: None,
                prompt: String::from("first"),
                response_mode: LlmResponseMode::StructuredJson,
                tools: Vec::new(),
            })
            .expect("first request should dispatch");
        thread::sleep(Duration::from_millis(20));
        bridge
            .send(LlmRequest {
                agent: String::from("alpha"),
                conversation_id: Some(String::from("thread")),
                world_override: None,
                prompt: String::from("second"),
                response_mode: LlmResponseMode::StructuredJson,
                tools: Vec::new(),
            })
            .expect("second request should dispatch");

        let outputs = wait_for_outputs(&bridge, 1, Duration::from_secs(2));
        let response = match outputs.into_iter().next().expect("response expected") {
            BridgeOutput::Response(turn) => turn.response,
            other => panic!("unexpected output: {other:?}"),
        };
        assert!(response.contains("second"));

        let conversation = bridge
            .conversation_state("alpha", "thread")
            .expect("conversation state should load")
            .expect("conversation should exist");
        assert_eq!(
            conversation
                .entries
                .iter()
                .filter(|entry| entry.kind == llm::LlmConversationEntryKind::Prompt)
                .count(),
            1
        );
    }

    #[test]
    fn newer_generated_conversation_request_supersedes_older_request() {
        let (bridge, _records) = stub_bridge(Duration::from_millis(150));

        bridge
            .handle_generation_command(LlmConversationGenerationCommand::GenerateConversation {
                session_id: String::from("session-1"),
                participants: vec![String::from("alpha"), String::from("bravo")],
                initial_message: String::from("first"),
                facts: vec![String::from("first fact")],
            })
            .expect("first request should dispatch");
        thread::sleep(Duration::from_millis(20));
        bridge
            .handle_generation_command(LlmConversationGenerationCommand::GenerateConversation {
                session_id: String::from("session-1"),
                participants: vec![String::from("alpha"), String::from("bravo")],
                initial_message: String::from("second"),
                facts: vec![String::from("second fact")],
            })
            .expect("second request should dispatch");

        let outputs = wait_for_outputs(&bridge, 1, Duration::from_secs(2));
        match outputs.into_iter().next().expect("event expected") {
            BridgeOutput::ConversationEvent(
                LlmConversationGenerationEvent::ConversationGenerated { conversation },
            ) => {
                assert_eq!(conversation.session_id, "session-1");
                assert_eq!(conversation.utterances.len(), 2);
            }
            other => panic!("unexpected output: {other:?}"),
        }
    }

    #[test]
    fn generated_conversation_failure_emits_failure_event() {
        let outputs = super::LlmRuntimeBridge::handle_generated_conversation(
            &stub_bridge(Duration::from_millis(0)).0,
            super::ConversationGenerationResult {
                session_id: String::from("session-1"),
                generation: 1,
                result: Err(String::from("invalid conversation JSON")),
            },
        );
        assert!(outputs.is_empty());

        let (bridge, _) = stub_bridge(Duration::from_millis(0));
        bridge
            .generated_conversations
            .lock()
            .expect("generated conversation lock")
            .insert(
                String::from("session-1"),
                super::ConversationGenerationState {
                    participants: vec![String::from("alpha")],
                    generation: 1,
                },
            );

        let outputs = bridge.handle_generated_conversation(super::ConversationGenerationResult {
            session_id: String::from("session-1"),
            generation: 1,
            result: Err(String::from("invalid conversation JSON")),
        });
        assert_eq!(
            outputs,
            vec![BridgeOutput::ConversationEvent(
                LlmConversationGenerationEvent::ConversationGenerationFailed {
                    session_id: String::from("session-1"),
                    reason: String::from("invalid conversation JSON"),
                }
            )]
        );
    }
}

/// Add the inbox and drain system for a specific action event type.
pub fn install_llm_actions<E: Event>(app: &mut App, inbox: LLMActionInbox<E>) {
    app.add_event::<E>()
        .insert_resource(inbox)
        .add_systems(PreUpdate, drain_llm_actions::<E>);
}

/// Drain pending actions from the inbox and publish them as Bevy events.
pub fn drain_llm_actions<E: Event>(inbox: Res<LLMActionInbox<E>>, mut writer: EventWriter<E>) {
    for action in inbox.drain() {
        writer.send(action);
    }
}

/// Example application that demonstrates the generated API.
pub fn run_example_app() {
    #[derive(Debug, Clone, Event, Serialize, Deserialize, schemars::JsonSchema, LLMActions)]
    enum DemoAction {
        /// Spawn a new enemy in the world.
        SpawnEnemy {
            /// X coordinate in world space.
            x: f32,
            /// Y coordinate in world space.
            y: f32,
        },

        /// Add score to the player.
        AddScore {
            /// Amount to add.
            amount: u32,
        },

        /// Toggle debug mode.
        ToggleDebug,
    }

    #[derive(Resource, Default)]
    struct DemoState {
        enemies_spawned: usize,
        score: u32,
        debug: bool,
    }

    fn apply_actions(mut state: ResMut<DemoState>, mut events: EventReader<DemoAction>) {
        for event in events.read() {
            match event {
                DemoAction::SpawnEnemy { .. } => {
                    state.enemies_spawned += 1;
                }
                DemoAction::AddScore { amount } => {
                    state.score += amount;
                }
                DemoAction::ToggleDebug => {
                    state.debug = !state.debug;
                }
            }
        }
    }

    let (sender, inbox) = llm_action_channel::<DemoAction>();

    let tool_names = DemoAction::llm_tool_names();
    let tool_defs = DemoAction::llm_tool_definitions();
    let toolset = DemoAction::llm_tool_set(sender.clone());
    let mut app = App::new();

    install_llm_actions(&mut app, inbox);
    app.insert_resource(DemoState::default());
    app.add_systems(Update, apply_actions);

    sender
        .send(DemoAction::SpawnEnemy { x: 2.0, y: 3.0 })
        .expect("demo action channel should be open");
    sender
        .send(DemoAction::AddScore { amount: 7 })
        .expect("demo action channel should be open");
    sender
        .send(DemoAction::ToggleDebug)
        .expect("demo action channel should be open");

    app.update();

    let state = app.world().resource::<DemoState>();
    println!(
        "llm actions ready: tools={:?} tool_count={} score={} enemies={} debug={}",
        tool_names,
        tool_defs.len(),
        state.score,
        state.enemies_spawned,
        state.debug
    );
    println!(
        "generated toolset contains spawn_enemy: {}",
        toolset.contains("spawn_enemy")
    );
    for def in tool_defs {
        println!("tool: {} -> {}", def.name, def.description);
    }
}
