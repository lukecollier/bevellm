use bevellm::{LLMActions, LlmRequest, LlmResponseMode, install_llm_actions, llm_action_channel};
use bevy_app::{App, Startup, Update};
use bevy_ecs::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Event, Serialize, Deserialize, JsonSchema, LLMActions)]
enum GameAction {
    /// Spawn a new enemy in the world.
    SpawnEnemy {
        /// X coordinate in world space.
        x: f32,
        /// Y coordinate in world space.
        y: f32,
    },

    /// Add gold to the player's inventory.
    AddGold {
        /// Amount of gold to add.
        amount: u32,
    },

    /// Toggle debug mode.
    ToggleDebug,
}

#[derive(Resource, Default)]
struct GameState {
    enemies_spawned: usize,
    gold: u32,
    debug: bool,
}

fn apply_actions(mut state: ResMut<GameState>, mut events: EventReader<GameAction>) {
    for event in events.read() {
        match event {
            GameAction::SpawnEnemy { .. } => {
                state.enemies_spawned += 1;
            }
            GameAction::AddGold { amount } => {
                state.gold += amount;
            }
            GameAction::ToggleDebug => {
                state.debug = !state.debug;
            }
        }
    }
}

fn apply_llm_requests(mut requests: EventReader<LlmRequest>) {
    for request in requests.read() {
        println!(
            "llm request -> agent={} prompt={}",
            request.agent, request.prompt
        );
    }
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .try_init();

    let (sender, inbox) = llm_action_channel::<GameAction>();
    let mut app = App::new();

    install_llm_actions(&mut app, inbox);
    app.add_event::<LlmRequest>();
    app.insert_resource(GameState::default());
    app.add_systems(Update, apply_actions);
    app.add_systems(Update, apply_llm_requests);
    app.add_systems(Startup, |mut requests: EventWriter<LlmRequest>| {
        requests.send(LlmRequest {
            agent: String::from("overworld_npc"),
            conversation_id: None,
            world_override: None,
            prompt: String::from("What is the next objective?"),
            response_mode: LlmResponseMode::StructuredJson,
            tools: Vec::new(),
        });
    });

    println!("tool names: {:?}", GameAction::llm_tool_names());
    for def in GameAction::llm_tool_definitions() {
        println!("tool: {} -> {}", def.name, def.description);
        println!("{}", serde_json::to_string_pretty(&def.parameters).unwrap());
    }

    let toolset = GameAction::llm_tool_set(sender.clone());
    println!(
        "toolset contains spawn_enemy: {}",
        toolset.contains("spawn_enemy")
    );

    sender
        .send(GameAction::SpawnEnemy { x: 1.5, y: 2.5 })
        .expect("action channel should be open");
    sender
        .send(GameAction::AddGold { amount: 25 })
        .expect("action channel should be open");
    sender
        .send(GameAction::ToggleDebug)
        .expect("action channel should be open");

    app.update();
    app.update();

    let state = app.world().resource::<GameState>();
    println!(
        "applied actions -> enemies={} gold={} debug={}",
        state.enemies_spawned, state.gold, state.debug
    );
}
