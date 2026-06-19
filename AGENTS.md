# Agent Guide

Concise startup context for coding agents working on this repo.

## Project Shape

`bevellm` is a Rust 2024 crate that connects Bevy ECS apps to local LLM workers. The public API is exported mostly from `src/lib.rs`; low-level model/runtime behavior lives in `src/llm.rs`; generated conversation playback and fact bookkeeping live in `src/conversation.rs`; the `LLMActions` derive macro lives in `crates/bevellm_macros`.

The runtime backend is Mistral.rs. This crate should not directly depend on or import Candle, even though Mistral.rs may use Candle transitively.

## Key Concepts

- `LlmRuntimePlugin` installs the background worker bridge into a Bevy app.
- `LlmRuntimeConfig` contains named `LlmRuntimeProfileConfig`s plus `LlmTaskRoutingConfig`.
- Routing currently separates turn generation from fact extraction.
- `LlmRequest` is the main Bevy-facing request event.
- `LlmResponse` is the main response event.
- Conversations are keyed by `agent` and optional `conversation_id`.
- Generated conversations use `LlmConversationGenerationCommand` and emit `LlmConversationGenerationEvent`.
- `PlannedUtterance.text` must be spoken dialogue only; action-only intent should use `tool_calls`.
- `LLMActions` derives local tool definitions from Bevy event enums.

## Tool Calling

Tool definitions are local types: `LlmToolDefinition`, `LlmToolSet`, and `LlmToolCall`. Do not reintroduce `rig-core`.

`LlmToolCallingMode::Auto` resolves by model:

- `SmolLM3_3BQ4KM` -> `AgenticXml`
- current Qwen and SmolLM2 models -> `Native`

Generated conversation prompts should mention the resolved active tool format so models know how proposed actions should be represented.

## Runtime Notes

Mistral.rs calls are async, but the Bevy boundary is synchronous. Worker threads own runtimes/executors and bridge async model calls internally.

Apple Silicon support is expected through the `mistralrs` `metal` feature. If local builds fail because Metal kernels cannot precompile, use:

```sh
env MISTRALRS_METAL_PRECOMPILE=0 cargo check --offline
```

This skips Metal precompile during build; it does not intentionally disable Metal support.

## Common Commands

Prefer offline checks when dependencies are already cached:

```sh
cargo fmt --check
cargo check --offline
cargo test --offline --lib
cargo check --offline --example conversation
cargo check --offline --example strategy
cargo check --offline --example demo
```

Run `cargo fmt` after Rust edits.

## Important Files

- `src/lib.rs`: public API, Bevy bridge, request routing, generated conversation prompt/parser, tests.
- `src/llm.rs`: Mistral model loading, prompt construction, response parsing, facts, conversation memory.
- `src/conversation.rs`: generated conversation playback and fact extraction/storage flow.
- `crates/bevellm_macros/src/lib.rs`: `LLMActions` derive macro.
- `examples/conversation.rs`: two-round generated conversation flow.
- `examples/strategy.rs`: strategy-style LLM example.
- `examples/demo.rs`: lightweight macro/action demo.

## Editing Guidance

- Keep the Bevy-facing API stable unless a task explicitly asks for a breaking change.
- Preserve model/profile configuration surfaces when changing runtime internals.
- Add focused tests for routing, prompt contracts, parsing, and fact flow changes.
- Do not revert unrelated user changes in the working tree.
- Use `rg` for search and `apply_patch` for manual edits.
