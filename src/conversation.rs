use crate::{
    GeneratedConversation, LlmConversationFact, LlmConversationGenerationEvent,
    LlmFactExtractionCommand, LlmFactExtractionEvent, LlmFactStoreCommand, LlmFactStoreEvent,
};
use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;
use log::{debug, error, info};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

const WORDS_PER_SECOND: f32 = 2.7;
const COMMA_PAUSE_SECS: f32 = 0.15;
const SENTENCE_PAUSE_SECS: f32 = 0.30;
const MIN_UTTERANCE_SECS: f32 = 0.8;
const MAX_UTTERANCE_SECS: f32 = 6.0;

/// Playback phase for a generated transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationTranscriptPhase {
    WaitingForConversation,
    ReplayingTranscript,
    CollectingFacts,
    Complete,
}

/// Current state of a generated conversation round.
#[derive(Debug, Clone, Resource)]
pub struct ConversationTranscriptState {
    pub phase: ConversationTranscriptPhase,
    pub session_id: Option<String>,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
}

impl Default for ConversationTranscriptState {
    fn default() -> Self {
        Self {
            phase: ConversationTranscriptPhase::WaitingForConversation,
            session_id: None,
            started_at: None,
            completed_at: None,
        }
    }
}

/// Playback queue for a generated transcript.
#[derive(Debug, Clone, Resource, Default)]
pub struct ConversationPlaybackState {
    pub session_id: Option<String>,
    pub participants: Vec<String>,
    pub started_at: Option<Instant>,
    pub queue: VecDeque<ScheduledUtterance>,
    pub pending_fact_store_requests: usize,
    pub pending_fact_extraction_requests: usize,
}

/// Cached facts attributed to each participant.
#[derive(Debug, Clone, Resource, Default)]
pub struct ParticipantFactLedger {
    pub facts_by_agent: HashMap<String, Vec<LlmConversationFact>>,
}

/// A scheduled transcript utterance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledUtterance {
    pub speaker: String,
    pub text: String,
    pub starts_after: Duration,
    pub duration: Duration,
}

/// Plugin that wires transcript playback and fact bookkeeping into Bevy.
#[derive(Debug, Default)]
pub struct ConversationFlowPlugin;

impl Plugin for ConversationFlowPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConversationTranscriptState>()
            .init_resource::<ConversationPlaybackState>()
            .init_resource::<ParticipantFactLedger>()
            .add_systems(
                Update,
                (
                    handle_generated_conversation_events,
                    record_fact_extraction_events,
                    record_fact_store_events,
                    play_generated_conversation,
                )
                    .chain(),
            );
    }
}

/// Build a normalized playback queue from a generated conversation.
pub fn build_playback_queue(conversation: &GeneratedConversation) -> VecDeque<ScheduledUtterance> {
    let mut starts_after = Duration::ZERO;
    let mut queue = VecDeque::with_capacity(conversation.utterances.len());
    for utterance in &conversation.utterances {
        if utterance.text.trim().is_empty() {
            continue;
        }
        let duration = estimate_utterance_duration(&utterance.text);
        queue.push_back(ScheduledUtterance {
            speaker: utterance.speaker.clone(),
            text: utterance.text.clone(),
            starts_after,
            duration,
        });
        starts_after += duration;
    }
    queue
}

/// Estimate the local playback duration for an utterance.
pub fn estimate_utterance_duration(text: &str) -> Duration {
    let word_count = text.split_whitespace().count().max(1) as f32;
    let commas = text.matches(',').count() as f32;
    let sentence_stops = text
        .chars()
        .filter(|ch| matches!(ch, '.' | '!' | '?'))
        .count() as f32;
    let seconds = (word_count / WORDS_PER_SECOND)
        + (commas * COMMA_PAUSE_SECS)
        + (sentence_stops * SENTENCE_PAUSE_SECS);
    Duration::from_secs_f32(seconds.clamp(MIN_UTTERANCE_SECS, MAX_UTTERANCE_SECS))
}

/// Format the prompt text sent to fact storage for a single utterance.
pub fn format_fact_store_prompt(utterance: &crate::PlannedUtterance) -> String {
    format!("**{}:** {}", utterance.speaker, utterance.text)
}

/// Print the current fact ledger.
pub fn print_fact_ledger(ledger: &ParticipantFactLedger) {
    for (agent, facts) in &ledger.facts_by_agent {
        info!("[facts] ledger agent={} count={}", agent, facts.len());
        for fact in facts {
            info!(
                "  - fact={} importance={:.2} expiry={}",
                fact.fact, fact.importance, fact.expiry
            );
        }
    }
}

fn handle_generated_conversation_events(
    mut transcript: ResMut<ConversationTranscriptState>,
    mut playback: ResMut<ConversationPlaybackState>,
    mut fact_extraction_commands: EventWriter<LlmFactExtractionCommand>,
    mut fact_store_commands: EventWriter<LlmFactStoreCommand>,
    mut responses: EventReader<LlmConversationGenerationEvent>,
) {
    for response in responses.read() {
        match response {
            LlmConversationGenerationEvent::ConversationGenerated { conversation } => {
                start_generated_conversation(
                    &mut transcript,
                    &mut playback,
                    conversation,
                    &mut fact_extraction_commands,
                    &mut fact_store_commands,
                );
            }
            LlmConversationGenerationEvent::ConversationGenerationFailed { session_id, reason } => {
                error!(
                    "[session] transcript generation failed -> session={} reason={}",
                    session_id, reason
                );
                transcript.phase = ConversationTranscriptPhase::Complete;
                transcript.session_id = Some(session_id.clone());
                transcript.completed_at = Some(Instant::now());
                playback.queue.clear();
                playback.pending_fact_store_requests = 0;
                playback.pending_fact_extraction_requests = 0;
                playback.session_id = Some(session_id.clone());
                playback.started_at = None;
                playback.participants.clear();
                debug!("[session] transcript generation failed reason={reason}");
            }
        }
    }
}

fn start_generated_conversation(
    transcript: &mut ConversationTranscriptState,
    playback: &mut ConversationPlaybackState,
    conversation: &GeneratedConversation,
    fact_extraction_commands: &mut EventWriter<LlmFactExtractionCommand>,
    fact_store_commands: &mut EventWriter<LlmFactStoreCommand>,
) {
    debug!(
        "[session] transcript ready -> session={} utterances={}",
        conversation.session_id,
        conversation.utterances.len()
    );
    transcript.phase = ConversationTranscriptPhase::ReplayingTranscript;
    transcript.session_id = Some(conversation.session_id.clone());
    transcript.started_at = Some(Instant::now());
    transcript.completed_at = None;

    playback.session_id = Some(conversation.session_id.clone());
    playback.participants = conversation.participants.clone();
    playback.started_at = Some(Instant::now());
    playback.queue = build_playback_queue(conversation);
    playback.pending_fact_store_requests = playback.queue.len();
    playback.pending_fact_extraction_requests = playback.queue.len();

    for (utterance_index, utterance) in conversation.utterances.iter().cloned().enumerate() {
        if utterance.text.trim().is_empty() {
            if !utterance.tool_calls.is_empty() {
                info!(
                    "[session] action utterance -> session={} speaker={} tool_calls={}",
                    conversation.session_id,
                    utterance.speaker,
                    utterance
                        .tool_calls
                        .iter()
                        .map(|call| call.tool.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            continue;
        }

        fact_extraction_commands.send(LlmFactExtractionCommand::ExtractFactFromUtterance {
            request_id: format!("{}:extract:{utterance_index}", conversation.session_id),
            session_id: conversation.session_id.clone(),
            utterance_index,
            utterance: utterance.clone(),
        });

        fact_store_commands.send(LlmFactStoreCommand::StoreFactsFromPrompt {
            request_id: format!("{}:fact:{utterance_index}", conversation.session_id),
            text: format_fact_store_prompt(&utterance),
            agents: conversation.participants.clone(),
        });
    }
}

fn record_fact_extraction_events(
    mut playback: ResMut<ConversationPlaybackState>,
    mut transcript: ResMut<ConversationTranscriptState>,
    mut events: EventReader<LlmFactExtractionEvent>,
) {
    for event in events.read() {
        match event {
            LlmFactExtractionEvent::FactsExtractedFromUtterance {
                session_id, facts, ..
            } => {
                debug!(
                    "[facts] extracted -> session={} count={}",
                    session_id,
                    facts.len()
                );
                playback.pending_fact_extraction_requests =
                    playback.pending_fact_extraction_requests.saturating_sub(1);
            }
            LlmFactExtractionEvent::FactExtractionFailedForUtterance {
                session_id, reason, ..
            } => {
                error!(
                    "[facts] extraction failed -> session={} reason={}",
                    session_id, reason
                );
                playback.pending_fact_extraction_requests =
                    playback.pending_fact_extraction_requests.saturating_sub(1);
            }
        }
    }

    update_completion_state(&mut transcript, &mut playback);
}

fn record_fact_store_events(
    mut transcript: ResMut<ConversationTranscriptState>,
    mut playback: ResMut<ConversationPlaybackState>,
    mut ledger: ResMut<ParticipantFactLedger>,
    mut events: EventReader<LlmFactStoreEvent>,
) {
    for event in events.read() {
        match event {
            LlmFactStoreEvent::FactsStored { agents, facts, .. } => {
                debug!("[facts] stored -> agents={agents:?} count={}", facts.len());
                ledger.merge_for_agents(agents, facts);
                playback.pending_fact_store_requests =
                    playback.pending_fact_store_requests.saturating_sub(1);
            }
            LlmFactStoreEvent::FactStorageFailed { agents, reason, .. } => {
                error!(
                    "[facts] storage failed -> agents={agents:?} reason={}",
                    reason
                );
                playback.pending_fact_store_requests =
                    playback.pending_fact_store_requests.saturating_sub(1);
            }
        }
    }

    update_completion_state(&mut transcript, &mut playback);
}

fn update_completion_state(
    transcript: &mut ConversationTranscriptState,
    playback: &mut ConversationPlaybackState,
) {
    if playback.session_id.is_none() {
        return;
    }

    if !playback.queue.is_empty() {
        transcript.phase = ConversationTranscriptPhase::ReplayingTranscript;
        return;
    }

    if playback.pending_fact_store_requests == 0 {
        transcript.phase = ConversationTranscriptPhase::Complete;
        transcript.completed_at = Some(Instant::now());
    } else {
        transcript.phase = ConversationTranscriptPhase::CollectingFacts;
    }
}

fn play_generated_conversation(
    mut transcript: ResMut<ConversationTranscriptState>,
    mut playback: ResMut<ConversationPlaybackState>,
) {
    if playback.started_at.is_none() {
        return;
    }

    let Some(started_at) = playback.started_at else {
        return;
    };
    let elapsed = started_at.elapsed();

    while playback
        .queue
        .front()
        .is_some_and(|utterance| elapsed >= utterance.starts_after)
    {
        let utterance = playback
            .queue
            .pop_front()
            .expect("queue front should exist");
        if utterance.text.is_empty() {
            info!(
                "[session] utterance -> session={} speaker={} start={:?} duration={:?} text=<empty>",
                playback.session_id.as_deref().unwrap_or("<unknown>"),
                utterance.speaker,
                utterance.starts_after,
                utterance.duration
            );
        } else {
            info!(
                "[session] utterance -> session={} speaker={} start={:?} duration={:?} text={}",
                playback.session_id.as_deref().unwrap_or("<unknown>"),
                utterance.speaker,
                utterance.starts_after,
                utterance.duration,
                utterance.text
            );
        }
    }

    update_completion_state(&mut transcript, &mut playback);
}

impl ParticipantFactLedger {
    pub fn merge_for_agents(&mut self, agents: &[String], facts: &[LlmConversationFact]) {
        for agent in agents {
            let stored = self.facts_by_agent.entry(agent.clone()).or_default();
            crate::llm::merge_conversation_facts(stored, facts.to_vec());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ConversationPlaybackState, ConversationTranscriptPhase, ConversationTranscriptState,
        ParticipantFactLedger, build_playback_queue, estimate_utterance_duration,
        format_fact_store_prompt,
    };
    use crate::{
        GeneratedConversation, LlmConversationFact, LlmConversationGenerationEvent,
        LlmFactExtractionCommand, LlmFactExtractionEvent, LlmFactStoreCommand, LlmFactStoreEvent,
        PlannedUtterance,
    };
    use bevy_app::{App, Update};
    use bevy_ecs::prelude::*;
    use std::time::Duration;

    #[derive(Resource, Default)]
    struct CapturedExtractionCommands(Vec<LlmFactExtractionCommand>);

    #[derive(Resource, Default)]
    struct CapturedStoreCommands(Vec<LlmFactStoreCommand>);

    fn capture_extractions(
        mut reader: EventReader<LlmFactExtractionCommand>,
        mut captured: ResMut<CapturedExtractionCommands>,
    ) {
        captured.0.extend(reader.read().cloned());
    }

    fn capture_store_commands(
        mut reader: EventReader<LlmFactStoreCommand>,
        mut captured: ResMut<CapturedStoreCommands>,
    ) {
        captured.0.extend(reader.read().cloned());
    }

    #[test]
    fn utterance_duration_is_deterministic_and_clamped() {
        let short = estimate_utterance_duration("Go.");
        let long = estimate_utterance_duration(
            "Scout the eastern ridge, check the bridge, and report back immediately after the sweep is complete.",
        );
        assert_eq!(short, estimate_utterance_duration("Go."));
        assert!(short >= Duration::from_secs_f32(super::MIN_UTTERANCE_SECS));
        assert!(long <= Duration::from_secs_f32(super::MAX_UTTERANCE_SECS));
        assert!(long > short);
    }

    #[test]
    fn playback_queue_uses_cumulative_start_offsets() {
        let conversation = GeneratedConversation {
            session_id: String::from("session-1"),
            participants: vec![String::from("alpha"), String::from("bravo")],
            utterances: vec![
                PlannedUtterance {
                    speaker: String::from("alpha"),
                    text: String::from("Hold the bridge."),
                    tool_calls: Vec::new(),
                },
                PlannedUtterance {
                    speaker: String::from("bravo"),
                    text: String::from("I see movement on the east path."),
                    tool_calls: Vec::new(),
                },
            ],
        };

        let queue = build_playback_queue(&conversation);
        let items = queue.into_iter().collect::<Vec<_>>();
        assert_eq!(items[0].starts_after, Duration::ZERO);
        assert_eq!(items[1].starts_after, items[0].duration);
    }

    #[test]
    fn playback_queue_skips_action_only_utterances() {
        let conversation = GeneratedConversation {
            session_id: String::from("session-1"),
            participants: vec![String::from("alpha")],
            utterances: vec![
                PlannedUtterance {
                    speaker: String::from("alpha"),
                    text: String::from(""),
                    tool_calls: vec![crate::LlmToolCall {
                        tool: String::from("advance_to_bridge"),
                        arguments: serde_json::json!({"direction":"east"}),
                    }],
                },
                PlannedUtterance {
                    speaker: String::from("alpha"),
                    text: String::from("Hold here."),
                    tool_calls: Vec::new(),
                },
            ],
        };

        let queue = build_playback_queue(&conversation);
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.front().unwrap().text, "Hold here.");
    }

    #[test]
    fn fact_store_prompt_uses_exact_utterance_text() {
        let utterance = PlannedUtterance {
            speaker: String::from("alpha"),
            text: String::from("Hold the bridge."),
            tool_calls: Vec::new(),
        };

        assert_eq!(
            format_fact_store_prompt(&utterance),
            "**alpha:** Hold the bridge."
        );
    }

    #[test]
    fn stored_facts_are_attributed_to_each_agent() {
        let mut ledger = ParticipantFactLedger::default();
        let facts = vec![LlmConversationFact {
            fact: String::from("Alpha ordered the bridge held."),
            importance: 0.85,
            expiry: String::from("unknown"),
        }];

        ledger.merge_for_agents(
            &[String::from("agent_alpha"), String::from("agent_bravo")],
            &facts,
        );

        assert_eq!(
            ledger
                .facts_by_agent
                .get("agent_alpha")
                .expect("agent alpha should have facts")
                .len(),
            1
        );
        assert_eq!(
            ledger
                .facts_by_agent
                .get("agent_bravo")
                .expect("agent bravo should have facts")
                .len(),
            1
        );
    }

    #[test]
    fn generation_event_dispatches_per_message_fact_commands() {
        let mut app = App::new();
        app.add_event::<LlmConversationGenerationEvent>()
            .add_event::<LlmFactExtractionCommand>()
            .add_event::<LlmFactStoreCommand>()
            .insert_resource(ConversationTranscriptState::default())
            .insert_resource(ConversationPlaybackState::default())
            .insert_resource(ParticipantFactLedger::default())
            .insert_resource(CapturedExtractionCommands::default())
            .insert_resource(CapturedStoreCommands::default())
            .add_systems(
                Update,
                (
                    super::handle_generated_conversation_events,
                    capture_extractions,
                    capture_store_commands,
                )
                    .chain(),
            );

        app.world_mut()
            .send_event(LlmConversationGenerationEvent::ConversationGenerated {
                conversation: GeneratedConversation {
                    session_id: String::from("session-1"),
                    participants: vec![String::from("alpha"), String::from("bravo")],
                    utterances: vec![PlannedUtterance {
                        speaker: String::from("alpha"),
                        text: String::from("Hold the bridge."),
                        tool_calls: Vec::new(),
                    }],
                },
            });

        app.update();

        let extraction = &app.world().resource::<CapturedExtractionCommands>().0;
        let store = &app.world().resource::<CapturedStoreCommands>().0;
        assert_eq!(extraction.len(), 1);
        assert_eq!(store.len(), 1);

        match &extraction[0] {
            LlmFactExtractionCommand::ExtractFactFromUtterance { utterance, .. } => {
                assert_eq!(utterance.text, "Hold the bridge.");
            }
        }

        match &store[0] {
            LlmFactStoreCommand::StoreFactsFromPrompt { text, agents, .. } => {
                assert_eq!(text, "**alpha:** Hold the bridge.");
                assert_eq!(agents, &vec![String::from("alpha"), String::from("bravo")]);
            }
        }
    }

    #[test]
    fn stored_facts_complete_only_after_queue_drains_and_storage_finishes() {
        let mut app = App::new();
        app.add_event::<LlmConversationGenerationEvent>()
            .add_event::<LlmFactExtractionCommand>()
            .add_event::<LlmFactStoreCommand>()
            .add_event::<LlmFactExtractionEvent>()
            .add_event::<LlmFactStoreEvent>()
            .insert_resource(ConversationTranscriptState::default())
            .insert_resource(ConversationPlaybackState::default())
            .insert_resource(ParticipantFactLedger::default())
            .add_systems(
                Update,
                (
                    super::handle_generated_conversation_events,
                    super::record_fact_extraction_events,
                    super::record_fact_store_events,
                    super::play_generated_conversation,
                )
                    .chain(),
            );

        app.world_mut()
            .send_event(LlmConversationGenerationEvent::ConversationGenerated {
                conversation: GeneratedConversation {
                    session_id: String::from("session-1"),
                    participants: vec![String::from("alpha"), String::from("bravo")],
                    utterances: vec![PlannedUtterance {
                        speaker: String::from("alpha"),
                        text: String::from("Hold the bridge."),
                        tool_calls: Vec::new(),
                    }],
                },
            });
        app.update();

        assert_eq!(
            app.world().resource::<ConversationTranscriptState>().phase,
            ConversationTranscriptPhase::CollectingFacts
        );

        app.world_mut().send_event(LlmFactStoreEvent::FactsStored {
            request_id: String::from("session-1:fact:0"),
            agents: vec![String::from("alpha"), String::from("bravo")],
            facts: vec![LlmConversationFact {
                fact: String::from("The bridge is being held."),
                importance: 0.9,
                expiry: String::from("unknown"),
            }],
        });
        app.update();

        assert_eq!(
            app.world().resource::<ConversationTranscriptState>().phase,
            ConversationTranscriptPhase::Complete
        );
    }
}
