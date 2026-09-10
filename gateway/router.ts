import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

const AI_CONTAINER_URL = 'http://sam-ai-router.flycast:8080';
const ROUTER_TOKEN = process.env.LLAMA_API_KEY!;

const provider = createOpenAICompatible({
    name: 'router',
    apiKey: ROUTER_TOKEN,
    baseURL: `${AI_CONTAINER_URL}/v1`,
});

export const routerModel = provider('plano-orchestrator-4b');
