use bevellm::{
    LLMAgent, LLMAgentWorld, LlmModel, LlmRequest, LlmResponse, LlmResponseMode, LlmRuntimeConfig,
    LlmRuntimePlugin, LlmRuntimeProfileConfig, LlmTaskRoutingConfig, LlmToolCallingMode,
    install_llm_world_sync, llm_world_context,
};
use bevy_app::{App, Startup, Update};
use bevy_ecs::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::{Duration, Instant};

const DEMO_AGENT_COUNT: usize = 4;

#[derive(Resource, Default)]
struct DemoStatus {
    text_responses_received: usize,
    started_at: Option<Instant>,
    strategy_prompt_sent: bool,
    responded_agents: HashSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct DemoWorldView {
    /// Lumber currently available to spend this turn.
    lumber: u32,
    /// Gold currently available to spend this turn.
    gold: u32,
    /// Population currently consuming housing.
    population_used: u32,
    /// Population capacity before housing is full.
    population_capacity: u32,
    /// Buildings already completed inside the settlement.
    current_buildings: Vec<String>,
    /// Structures that can be built next, with compact purpose tags.
    build_options: Vec<BuildOption>,
    /// Threat level posed by player 1 on a 0-10 scale.
    player_1_threat: u32,
    /// Compact summary of player 1's current posture.
    player_1_posture: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct BuildOption {
    /// Exact structure name that matches the build tool input.
    name: String,
    /// Lumber cost paid immediately when construction starts.
    lumber_cost: u32,
    /// Gold cost paid immediately when construction starts.
    gold_cost: u32,
    /// Population capacity added by this structure, if any.
    population_capacity_gain: u32,
    /// Short role tag such as housing, lumber, gold, or military.
    role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct BuildPlanningView {
    /// Lumber available to spend right now.
    lumber: u32,
    /// Gold available to spend right now.
    gold: u32,
    /// Population currently using housing.
    population_used: u32,
    /// Maximum population supported by current housing.
    population_capacity: u32,
    /// Buildings already completed.
    current_buildings: Vec<String>,
    /// Structures relevant to the next build decision.
    build_options: Vec<BuildPlanningOption>,
    /// Nearby enemy pressure from 0 to 10.
    player_1_threat: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct BuildPlanningOption {
    /// Exact structure name to use in tool calls.
    name: String,
    /// Lumber cost paid immediately.
    lumber_cost: u32,
    /// Gold cost paid immediately.
    gold_cost: u32,
    /// Short role label for the structure.
    role: String,
}

impl BuildPlanningView {
    fn from_world(world: &DemoWorldView) -> Self {
        Self {
            lumber: world.lumber,
            gold: world.gold,
            population_used: world.population_used,
            population_capacity: world.population_capacity,
            current_buildings: world.current_buildings.clone(),
            build_options: world
                .build_options
                .iter()
                .map(|option| BuildPlanningOption {
                    name: option.name.clone(),
                    lumber_cost: option.lumber_cost,
                    gold_cost: option.gold_cost,
                    role: option.role.clone(),
                })
                .collect(),
            player_1_threat: world.player_1_threat,
        }
    }
}

fn spawn_agent(mut commands: Commands) {
    for index in 0..DEMO_AGENT_COUNT {
        let agent_id = format!("agent_{}", index + 1);
        commands.spawn((
            LLMAgent {
                id: agent_id.clone(),
            },
            LLMAgentWorld(initial_world_view(index)),
        ));
    }
}

fn queue_strategy_request(
    agents: Query<(&LLMAgent, &LLMAgentWorld<DemoWorldView>)>,
    mut status: ResMut<DemoStatus>,
    mut requests: EventWriter<LlmRequest>,
) {
    if status.strategy_prompt_sent {
        return;
    }

    for (agent, world) in agents.iter() {
        status.started_at = status.started_at.or(Some(Instant::now()));
        let request_view = BuildPlanningView::from_world(&world.0);
        let world_override = match llm_world_context(&request_view) {
            Ok(world_override) => Some(world_override),
            Err(err) => {
                eprintln!(
                    "failed to build reduced world view for agent={}: {err}",
                    agent.id
                );
                None
            }
        };
        requests.send(LlmRequest {
            agent: agent.id.clone(),
            conversation_id: None,
            world_override,
            prompt: String::from(
                "Review this settlement world state and explain the best next move in one short plain-text sentence under 18 words. Do not call any tools.",
            ),
            response_mode: LlmResponseMode::StructuredJson,
            tools: Vec::new(),
        });
        println!("queued strategy analysis request for agent={}", agent.id);
    }

    status.strategy_prompt_sent = true;
}

fn handle_llm_responses(mut status: ResMut<DemoStatus>, mut responses: EventReader<LlmResponse>) {
    for response in responses.read() {
        println!("llm response -> agent={}", response.agent);
        status.responded_agents.insert(response.agent.clone());
        if let Some(text) = &response.response {
            status.text_responses_received += 1;
            println!("llm text -> {}", text);
        }
        for call in &response.tool_calls {
            println!(
                "llm tool -> agent={} tool={} args={}",
                response.agent, call.tool, call.arguments
            );
        }

        if status.responded_agents.len() >= DEMO_AGENT_COUNT {
            if let Some(started_at) = status.started_at {
                println!("received demo responses in {:?}", started_at.elapsed());
            }
        }
    }
}

fn initial_world_view(index: usize) -> DemoWorldView {
    let lumber = 55 + (index as u32 * 10);
    let gold = 45 + (index as u32 * 8);
    let population_used = 6 + (index as u32 % 3);
    let population_capacity = 8 + ((index as u32 / 2) * 2);
    let threat = 2 + index as u32;
    let posture = match index {
        0 => "scouting borders",
        1 => "fortifying river",
        2 => "raiding trade",
        _ => "massing troops",
    };

    DemoWorldView {
        lumber,
        gold,
        population_used,
        population_capacity,
        current_buildings: match index {
            0 => vec![String::from("town_hall"), String::from("farm")],
            1 => vec![String::from("town_hall"), String::from("house")],
            2 => vec![String::from("town_hall"), String::from("sawmill")],
            _ => vec![
                String::from("town_hall"),
                String::from("farm"),
                String::from("house"),
            ],
        },
        build_options: vec![
            BuildOption {
                name: String::from("house"),
                lumber_cost: 30,
                gold_cost: 10,
                population_capacity_gain: 4,
                role: String::from("housing"),
            },
            BuildOption {
                name: String::from("sawmill"),
                lumber_cost: 60,
                gold_cost: 20,
                population_capacity_gain: 0,
                role: String::from("lumber"),
            },
            BuildOption {
                name: String::from("market"),
                lumber_cost: 40,
                gold_cost: 60,
                population_capacity_gain: 0,
                role: String::from("gold"),
            },
            BuildOption {
                name: String::from("barracks"),
                lumber_cost: 80,
                gold_cost: 50,
                population_capacity_gain: 0,
                role: String::from("military"),
            },
        ],
        player_1_threat: threat,
        player_1_posture: String::from(posture),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .try_init();

    let mut app = App::new();
    app.add_plugins(LlmRuntimePlugin {
        config: LlmRuntimeConfig {
            profiles: vec![LlmRuntimeProfileConfig {
                id: String::from("qwen"),
                model: LlmModel::Qwen2_5_1_5BInstructQ4KM,
                use_gpu: true,
                tool_calling: LlmToolCallingMode::Auto,
                worker_count: 1,
                temperature: None,
                top_p: None,
                max_new_tokens: 512,
                ..Default::default()
            }],
            routing: LlmTaskRoutingConfig {
                turn_generation_profile: String::from("qwen"),
                fact_extraction_profile: String::from("qwen"),
            },
        },
    });
    install_llm_world_sync::<DemoWorldView>(&mut app);
    app.insert_resource(DemoStatus::default());
    app.add_systems(Startup, spawn_agent);
    app.add_systems(
        Update,
        (queue_strategy_request, handle_llm_responses).chain(),
    );

    let mut ticks = 0usize;
    loop {
        app.update();
        std::thread::sleep(Duration::from_millis(100));
        ticks += 1;

        let status = app.world().resource::<DemoStatus>();
        if status.responded_agents.len() >= DEMO_AGENT_COUNT {
            break;
        }

        if ticks.is_multiple_of(50) {
            println!("waiting for llm response...");
        }
    }

    app.update();
    println!(
        "final state -> responses={} agents={}",
        app.world().resource::<DemoStatus>().text_responses_received,
        DEMO_AGENT_COUNT
    );

    Ok(())
}
