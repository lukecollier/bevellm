use bevellm::conversation::{
    ConversationFlowPlugin, ConversationTranscriptPhase, ConversationTranscriptState,
    ParticipantFactLedger, print_fact_ledger,
};
use bevellm::{
    LLMAgent, LLMAgentWorld, LlmConversationCommand, LlmConversationGenerationCommand,
    LlmConversationGenerationEvent, LlmModel, LlmRuntimeConfig, LlmRuntimePlugin,
    LlmRuntimeProfileConfig, LlmTaskRoutingConfig, LlmToolCall, LlmToolCallingMode,
    install_llm_world_sync,
};
use bevy_app::{App, Startup, Update};
use bevy_ecs::prelude::*;
use bevy_transform::components::Transform;
use log::{debug, error, info};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

const INNER_DIALOGUE_ID: &str = "inner_dialogue";
const ALPHA_ID: &str = "agent_alpha";
const BRAVO_ID: &str = "agent_bravo";
const FIRST_SESSION_ID: &str = "alpha_bravo_patrol";
const SECOND_SESSION_ID: &str = "alpha_bravo_patrol_followup";
const HEARING_RANGE: f32 = 3.0;
const PATROL_ACTION_TOOL_SCHEMA: &str = r#"Available patrol action tools for proposed actions:
- warn_about_exposed_path {"path":"eastern"|"western"|"bridge","severity":"low"|"medium"|"high"}
- hold_position {"agent":"agent_alpha"|"agent_bravo","reason":"short reason"}
- secure_bridge_approach {"leader":"agent_alpha"|"agent_bravo","approach":"east"|"west"|"center"}
- mark_sight_line {"path":"eastern"|"western"|"bridge","threat":"short description"}

When a participant wants to do one of these actions, put it in that utterance's tool_calls array.
If the utterance includes spoken dialogue, keep that dialogue in text and put the action in tool_calls.
Only leave text empty when the beat is purely an action with no spoken line.
Do not narrate action intent in text."#;

#[derive(Resource, Default)]
struct DemoStatus {
    started_at: Option<Instant>,
    first_round_requested: bool,
    second_round_requested: bool,
}

#[derive(Debug, Clone, Default, Resource)]
struct ToolCallProbe {
    calls_by_session: HashMap<String, Vec<SpeakerToolCall>>,
}

#[derive(Debug, Clone)]
struct SpeakerToolCall {
    speaker: String,
    call: LlmToolCall,
}

#[derive(Resource, Default)]
struct ProximityState {
    in_range_pairs: HashSet<PairKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PairKey {
    a: String,
    b: String,
}

impl PairKey {
    fn new(left: &str, right: &str) -> Self {
        if left <= right {
            Self {
                a: left.to_string(),
                b: right.to_string(),
            }
        } else {
            Self {
                a: right.to_string(),
                b: left.to_string(),
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct ConversationWorldView {
    agent_name: String,
    persona: String,
    current_goal: String,
    nearby_agents: Vec<NearbyAgent>,
    known_agents: Vec<KnownAgent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct NearbyAgent {
    id: String,
    distance: f32,
    disposition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct KnownAgent {
    id: String,
    is_nearby: bool,
    disposition: String,
    last_known_distance: f32,
    last_known_status: String,
}

#[derive(Debug, Clone, PartialEq)]
struct AgentSnapshot {
    entity: Entity,
    id: String,
    transform: Transform,
    world: ConversationWorldView,
}

fn spawn_agents(mut commands: Commands) {
    commands.spawn((
        LLMAgent {
            id: String::from(ALPHA_ID),
        },
        Transform::from_xyz(0.0, 0.0, 0.0),
        LLMAgentWorld(initial_world_view(
            "Commander Alpha",
            "Measured squad leader who gives concise, decisive instructions.",
            "Confirm the scout's report and secure the bridge approach before advancing.",
            BRAVO_ID,
            "friendly",
            "holding position near the bridge",
        )),
    ));

    commands.spawn((
        LLMAgent {
            id: String::from(BRAVO_ID),
        },
        Transform::from_xyz(2.0, 0.0, 0.0),
        LLMAgentWorld(initial_world_view(
            "Scout Bravo",
            "Alert reconnaissance scout who speaks plainly and prioritizes warnings.",
            "Warn the commander about the exposed eastern path before either of them moves.",
            ALPHA_ID,
            "friendly",
            "watching the eastern path",
        )),
    ));
}

fn initialize_conversation_threads(mut chat_commands: EventWriter<LlmConversationCommand>) {
    for agent in [ALPHA_ID, BRAVO_ID] {
        chat_commands.send(LlmConversationCommand::AppendMessage {
            agent: agent.to_string(),
            conversation_id: String::from(INNER_DIALOGUE_ID),
            message: String::from(
                "Private reasoning thread initialized for future action planning.",
            ),
        });
    }
}

fn initial_world_view(
    agent_name: &str,
    persona: &str,
    current_goal: &str,
    known_agent_id: &str,
    disposition: &str,
    last_known_status: &str,
) -> ConversationWorldView {
    ConversationWorldView {
        agent_name: agent_name.to_string(),
        persona: persona.to_string(),
        current_goal: current_goal.to_string(),
        nearby_agents: Vec::new(),
        known_agents: vec![KnownAgent {
            id: known_agent_id.to_string(),
            is_nearby: false,
            disposition: disposition.to_string(),
            last_known_distance: 999.0,
            last_known_status: last_known_status.to_string(),
        }],
    }
}

fn request_generated_conversation(
    session_id: &str,
    participants: Vec<String>,
    initial_message: String,
    facts: Vec<String>,
) -> LlmConversationGenerationCommand {
    LlmConversationGenerationCommand::GenerateConversation {
        session_id: session_id.to_string(),
        participants,
        initial_message,
        facts,
    }
}

fn sync_proximity_world_and_chat(
    mut param_set: ParamSet<(
        Query<(
            Entity,
            &LLMAgent,
            &Transform,
            &LLMAgentWorld<ConversationWorldView>,
        )>,
        Query<&mut LLMAgentWorld<ConversationWorldView>>,
    )>,
    mut proximity: ResMut<ProximityState>,
    mut demo: ResMut<DemoStatus>,
    mut commands: EventWriter<LlmConversationGenerationCommand>,
) {
    let snapshots = {
        let agents = param_set.p0();
        agents
            .iter()
            .map(|(entity, agent, transform, world)| AgentSnapshot {
                entity,
                id: agent.id.clone(),
                transform: *transform,
                world: world.0.clone(),
            })
            .collect::<Vec<_>>()
    };

    let mut nearby_by_entity = HashMap::<Entity, Vec<NearbyAgent>>::new();
    let mut known_by_entity = HashMap::<Entity, Vec<KnownAgent>>::new();
    let mut next_in_range = HashSet::<PairKey>::new();

    for snapshot in &snapshots {
        known_by_entity.insert(snapshot.entity, snapshot.world.known_agents.clone());
    }

    for left_index in 0..snapshots.len() {
        for right_index in (left_index + 1)..snapshots.len() {
            let left = &snapshots[left_index];
            let right = &snapshots[right_index];
            let distance = distance_between(&left.transform, &right.transform);

            update_known_agent(
                known_by_entity
                    .get_mut(&left.entity)
                    .expect("left known agent entry"),
                &right.id,
                distance,
                distance <= HEARING_RANGE,
            );
            update_known_agent(
                known_by_entity
                    .get_mut(&right.entity)
                    .expect("right known agent entry"),
                &left.id,
                distance,
                distance <= HEARING_RANGE,
            );

            if distance <= HEARING_RANGE {
                next_in_range.insert(PairKey::new(&left.id, &right.id));
                nearby_by_entity
                    .entry(left.entity)
                    .or_default()
                    .push(NearbyAgent {
                        id: right.id.clone(),
                        distance,
                        disposition: known_disposition(&left.world, &right.id),
                    });
                nearby_by_entity
                    .entry(right.entity)
                    .or_default()
                    .push(NearbyAgent {
                        id: left.id.clone(),
                        distance,
                        disposition: known_disposition(&right.world, &left.id),
                    });
            }
        }
    }

    let entering_pairs = next_in_range
        .difference(&proximity.in_range_pairs)
        .cloned()
        .collect::<Vec<_>>();
    if !entering_pairs.is_empty() {
        demo.started_at.get_or_insert_with(|| Instant::now());
    }

    for pair in &entering_pairs {
        if pair == &PairKey::new(ALPHA_ID, BRAVO_ID) && !demo.first_round_requested {
            commands.send(request_generated_conversation(
                FIRST_SESSION_ID,
                vec![String::from(ALPHA_ID), String::from(BRAVO_ID)],
                format!(
                    "Scout Bravo, report in. What's the status of the eastern path?\n\n\
This is a tool-call extraction probe: if either participant wants to warn, hold, mark, or secure, keep the spoken line in text and represent the action with tool_calls in the same utterance when possible.\n\n\
{PATROL_ACTION_TOOL_SCHEMA}"
                ),
                vec![
                    String::from("The eastern path is exposed to enemy sight lines."),
                    String::from("The bridge approach must be secured before the squad advances."),
                    String::from("Scout Bravo should warn about the exposed eastern path."),
                    String::from("Commander Alpha should choose whether to hold position or secure the bridge approach."),
                ],
            ));
            debug!("[demo] requested generated conversation {FIRST_SESSION_ID}");
            demo.first_round_requested = true;
        }
    }

    {
        let mut worlds = param_set.p1();
        for snapshot in &snapshots {
            if let Ok(mut world) = worlds.get_mut(snapshot.entity) {
                let next_nearby_agents = nearby_by_entity
                    .remove(&snapshot.entity)
                    .unwrap_or_default();
                if world.nearby_agents != next_nearby_agents {
                    world.nearby_agents = next_nearby_agents;
                }
                if let Some(known_agents) = known_by_entity.remove(&snapshot.entity) {
                    if world.known_agents != known_agents {
                        world.known_agents = known_agents;
                    }
                }
            }
        }
    }

    proximity.in_range_pairs = next_in_range;
}

fn collect_fact_strings(ledger: &ParticipantFactLedger) -> Vec<String> {
    let mut facts = Vec::new();
    for agent in [ALPHA_ID, BRAVO_ID] {
        if let Some(agent_facts) = ledger.facts_by_agent.get(agent) {
            facts.extend(agent_facts.iter().map(|fact| fact.fact.clone()));
        }
    }
    facts
}

fn observe_generated_tool_calls(
    mut events: EventReader<LlmConversationGenerationEvent>,
    mut probe: ResMut<ToolCallProbe>,
) {
    for event in events.read() {
        match event {
            LlmConversationGenerationEvent::ConversationGenerated { conversation } => {
                let mut session_calls = Vec::new();
                for utterance in &conversation.utterances {
                    for call in &utterance.tool_calls {
                        info!(
                            "[demo] generated tool call -> session={} speaker={} tool={} arguments={}",
                            conversation.session_id, utterance.speaker, call.tool, call.arguments
                        );
                        session_calls.push(SpeakerToolCall {
                            speaker: utterance.speaker.clone(),
                            call: call.clone(),
                        });
                    }
                }

                info!(
                    "[demo] tool-call probe session={} calls={}",
                    conversation.session_id,
                    session_calls.len()
                );
                probe
                    .calls_by_session
                    .insert(conversation.session_id.clone(), session_calls);
            }
            LlmConversationGenerationEvent::ConversationGenerationFailed { session_id, .. } => {
                probe
                    .calls_by_session
                    .insert(session_id.clone(), Vec::new());
            }
        }
    }
}

fn print_tool_call_probe(probe: &ToolCallProbe) {
    let total_calls = probe.calls_by_session.values().map(Vec::len).sum::<usize>();
    info!("[demo] tool-call probe total calls={total_calls}");

    for (session_id, calls) in &probe.calls_by_session {
        info!(
            "[demo] tool-call probe summary -> session={} calls={}",
            session_id,
            calls.len()
        );
        for call in calls {
            info!(
                "  - speaker={} tool={} arguments={}",
                call.speaker, call.call.tool, call.call.arguments
            );
        }
    }
}

fn update_known_agent(
    known_agents: &mut [KnownAgent],
    other_id: &str,
    distance: f32,
    is_nearby: bool,
) {
    for known_agent in known_agents {
        if known_agent.id == other_id {
            known_agent.last_known_distance = distance;
            known_agent.is_nearby = is_nearby;
            known_agent.last_known_status = if is_nearby {
                String::from("within hearing range")
            } else {
                String::from("out of hearing range")
            };
        }
    }
}

fn known_disposition(world: &ConversationWorldView, other_id: &str) -> String {
    world
        .known_agents
        .iter()
        .find(|agent| agent.id == other_id)
        .map(|agent| agent.disposition.clone())
        .unwrap_or_else(|| String::from("unknown"))
}

fn distance_between(a: &Transform, b: &Transform) -> f32 {
    a.translation.distance(b.translation)
}

fn print_world_snapshot(agent: &str, world: &ConversationWorldView) {
    match serde_json::to_string_pretty(world) {
        Ok(json) => debug!("[demo] world snapshot for {agent}:\n{json}"),
        Err(err) => error!("[demo] failed to serialize world snapshot for {agent}: {err}"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();

    let mut app = App::new();
    app.add_plugins(LlmRuntimePlugin {
        config: LlmRuntimeConfig {
            profiles: vec![
                LlmRuntimeProfileConfig {
                    id: String::from("qwen"),
                    model: LlmModel::Qwen2_5_1_5BInstructQ4KM,
                    use_gpu: true,
                    tool_calling: LlmToolCallingMode::Auto,
                    worker_count: 1,
                    temperature: None,
                    top_p: None,
                    max_new_tokens: 1024,
                    ..Default::default()
                },
                LlmRuntimeProfileConfig {
                    id: String::from("smollm2"),
                    model: LlmModel::SmolLM2_360MInstruct,
                    use_gpu: true,
                    tool_calling: LlmToolCallingMode::Auto,
                    worker_count: 1,
                    temperature: None,
                    top_p: None,
                    max_new_tokens: 128,
                    ..Default::default()
                },
            ],
            routing: LlmTaskRoutingConfig {
                turn_generation_profile: String::from("qwen"),
                fact_extraction_profile: String::from("smollm2"),
            },
        },
    });
    app.add_plugins(ConversationFlowPlugin);
    install_llm_world_sync::<ConversationWorldView>(&mut app);
    app.insert_resource(DemoStatus::default());
    app.insert_resource(ProximityState::default());
    app.insert_resource(ToolCallProbe::default());
    app.add_systems(
        Startup,
        (spawn_agents, initialize_conversation_threads).chain(),
    );
    app.add_systems(
        Update,
        (sync_proximity_world_and_chat, observe_generated_tool_calls),
    );

    {
        let mut worlds = app
            .world_mut()
            .query::<(&LLMAgent, &LLMAgentWorld<ConversationWorldView>)>();
        for (agent, world) in worlds.iter(app.world()) {
            print_world_snapshot(&agent.id, world);
        }
    }

    let mut last_phase = ConversationTranscriptPhase::WaitingForConversation;
    let mut last_session_id = None::<String>;

    loop {
        app.update();

        let (phase, session_id, started_at, completed_at) = {
            let transcript = app.world().resource::<ConversationTranscriptState>();
            (
                transcript.phase,
                transcript.session_id.clone(),
                transcript.started_at,
                transcript.completed_at,
            )
        };

        if phase != last_phase || session_id != last_session_id {
            info!(
                "[demo] transcript phase={:?} session={:?} started_at={:?} completed_at={:?}",
                phase, session_id, started_at, completed_at
            );
            last_phase = phase;
            last_session_id = session_id.clone();
        }

        let first_round_complete = phase == ConversationTranscriptPhase::Complete
            && session_id.as_deref() == Some(FIRST_SESSION_ID);
        let second_round_complete = phase == ConversationTranscriptPhase::Complete
            && session_id.as_deref() == Some(SECOND_SESSION_ID);

        if first_round_complete && !app.world().resource::<DemoStatus>().second_round_requested {
            let ledger = app.world().resource::<ParticipantFactLedger>();
            let followup_facts = collect_fact_strings(ledger);
            app.world_mut().send_event(request_generated_conversation(
                SECOND_SESSION_ID,
                vec![String::from(ALPHA_ID), String::from(BRAVO_ID)],
                String::from("Scout Bravo, what's your report on the eastern path?"),
                followup_facts,
            ));
            app.world_mut()
                .resource_mut::<DemoStatus>()
                .second_round_requested = true;
            info!("[demo] requested follow-up conversation {SECOND_SESSION_ID}");
        }

        if second_round_complete {
            break;
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    let ledger = app.world().resource::<ParticipantFactLedger>();
    print_fact_ledger(ledger);
    let probe = app.world().resource::<ToolCallProbe>();
    print_tool_call_probe(probe);

    Ok(())
}
