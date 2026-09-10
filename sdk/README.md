# @wundero/ai-router

An AI SDK provider that automatically routes each request to the best model/provider through a backing gateway. You write a normal `generateText` / `streamText` call; the router decides which underlying model actually answers.

## What it does

`@wundero/ai-router` is a [Vercel AI SDK](https://ai-sdk.dev) language-model provider. It implements the AI SDK provider specification (`v2`, `v3`, and `v4`) and forwards every generation/stream request to a central gateway (`https://ai.1d.gg`).

The flow:

1. Your code calls `generateText({ model: samAI('glm-5.3'), ... })`.
2. The provider calls the gateway's `/decide` endpoint with the conversation.
3. A small orchestrator model classifies the intent (chat, coding, reasoning, vision, …) and picks a concrete model + provider.
4. The provider forwards the call to `/generate` or `/stream` with the chosen decision, and streams the result back.

The model id you pass to the factory is largely informational — the gateway decides the actual model based on the conversation. You can also pin a specific decision per-model or per-call (see below).

## Install

```bash
npm install @wundero/ai-router ai
# or: bun add / pnpm add / yarn add
```

Runs on Bun, Deno, or Node. Supports both ESM and CommonJS. Requires TypeScript.

## Quick start

```ts
import { generateText } from 'ai';
import { createSamAI } from '@wundero/ai-router';

const samAI = createSamAI({
  apiKey: process.env.SAM_AI_API_KEY,
});

const { text } = await generateText({
  model: samAI('glm-5.3'),
  prompt: 'Explain the difference between a gateway and a proxy.',
});

console.log(text);
```

Streaming works the same way:

```ts
import { streamText } from 'ai';
import { createSamAI } from '@wundero/ai-router';

const samAI = createSamAI({ apiKey: process.env.SAM_AI_API_KEY });

const result = streamText({
  model: samAI('glm-5.3'),
  prompt: 'Write a short poem about routing.',
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

## API

### `createSamAI(options)` — v4 (default)

The default export targets the AI SDK v4 provider spec. Imported from the package root or `@wundero/ai-router/v4`.

### `createSamAIV3(options)` / `createSamAIV2(options)`

Older spec versions for compatibility with older `ai` package releases. Imported from `@wundero/ai-router/v3` and `@wundero/ai-router/v2`.

### Options

```ts
interface SamAIOptions {
  apiKey?: string;   // gateway API key; falls back to SAM_AI_API_KEY env var
  storage?: Storage; // unstorage instance for client-side decision caching
  workersAI?: Ai;    // Cloudflare Workers AI binding for edge execution
}
```

Authentication is read from the `SAM_AI_API_KEY` environment variable if `apiKey` is not provided.

### Per-model settings

```ts
interface CustomChatSettings {
  decision?: Decision; // pin { provider, model }
  sessionId?: string;  // pin a session id
}
```

```ts
// Pin a specific model for this language-model instance:
const model = samAI('glm-5.3', {
  decision: { provider: 'deepseek', model: 'deepseek-v4-pro' },
});
```

`Decision` has shape `{ model: string; provider: 'cloudflare' | 'deepseek' | 'moonshot' | 'zhipu' }`.

## Sessions & caching

Routing decisions are keyed to a session id and cached (gateway-side in Redis, 12h TTL). Consecutive turns in the same conversation reuse the decision instead of re-running the orchestrator.

Pass a session id per call via provider options:

```ts
generateText({
  model: samAI('glm-5.3'),
  prompt: '...',
  providerOptions: {
    session: { id: 'my-session-123' },
  },
});
```

Or supply a `storage` (unstorage) instance so decisions are cached locally too:

```ts
import { createStorage } from 'unstorage';
import redisDriver from 'unstorage/drivers/redis';

const samAI = createSamAI({
  apiKey: process.env.SAM_AI_API_KEY,
  storage: createStorage({ driver: redisDriver({ url: process.env.REDIS_URL! }) }),
});
```

## Cloudflare Workers edge execution

If you pass a Workers AI binding, requests that route to the `cloudflare` provider run directly on the edge (no round-trip to the gateway):

```ts
import { createSamAI } from '@wundero/ai-router';

export default {
  async fetch(request, env) {
    const samAI = createSamAI({
      apiKey: env.SAM_AI_API_KEY,
      workersAI: env.AI, // Workers AI binding
    });

    const { text } = await generateText({
      model: samAI('glm-5.3'),
      prompt: 'Hello from the edge.',
    });

    return Response.json({ text });
  },
};
```

## Backing gateway implementation

The SDK is a thin client over an HTTP gateway (Bun, `Bun.serve`). It exposes three routes:

| Route                | Purpose                                                        |
| -------------------- | -------------------------------------------------------------- |
| `POST /decide`       | Classify intent and return a `{ model, provider }` decision.   |
| `POST /:v/generate`  | Non-streaming generation for spec `v2`/`v3`/`v4`.              |
| `POST /:v/stream`    | NDJSON streaming for spec `v2`/`v3`/`v4`.                      |

All routes require `Authorization: Bearer <token>`.

### Decision flow (`/decide`)

1. A small **orchestrator model** (`plano-orchestrator-4b`, served from a separate `llm` container via an OpenAI-compatible endpoint) reads the conversation and picks one or more "routes" from a fixed set: `general_chat`, `creative_writing`, `deep_reasoning`, `text_transformation`, `vision_multimodal`, `tool_use_agentic`, `coding`, `agentic_coding`.
2. Each route maps to a **shared model** (e.g. `glm-5.3-flash` for fast chat, `glm-5.3` for deep reasoning/creative work).
3. The shared model resolves to a **concrete provider + model id** through a fallback chain:
   - `glm-*` → Zhipu (Z.ai), fallback Cloudflare
   - `deepseek-*` → DeepSeek, fallback Cloudflare
   - `kimi-*` → Moonshot, fallback Cloudflare
4. The decision is cached in Redis under `route:<sessionId>` (12h TTL) so repeat turns skip the orchestrator.

### Providers

- **Cloudflare** — via `ai-gateway-provider` (`createAiGateway` + `createUnified`) on top of Cloudflare Workers AI.
- **DeepSeek** — `@ai-sdk/deepseek`.
- **Moonshot** (Kimi) — `@ai-sdk/moonshotai`.
- **Zhipu** (Z.ai) — `@ai-sdk/zai`.

Each provider is optional: if its API key isn't set, requests fall back to Cloudflare. This is what makes the SDK's `cloudflare` decision + `workersAI` binding path a pure edge execution with no upstream key needed.
