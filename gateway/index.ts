import { createStorage } from 'unstorage';
import redisDriver from 'unstorage/drivers/redis';
import { z, ZodError } from 'zod';
import type { BunRequest } from 'bun';
import { AISDKError, generateText } from 'ai';
import { decisionSchema, type Decision } from './types';
import { routerModel } from './router';
import { getDecisionForSharedModel, MODELS, provider } from './providers';
import type {
    LanguageModelV2StreamPart,
    LanguageModelV3StreamPart,
    LanguageModelV4CallOptions,
    LanguageModelV4StreamPart,
} from '@ai-sdk/provider';
import { COMPAT } from './versions';
import { getSchema } from './schemas';
const GATEWAY_TOKEN = process.env.GATEWAY_AUTH_TOKEN!;

const decideSchema = z.object({
    conversation: z.string(),
    session: z.string().nullish(),
});

export type DecideAPISchema = z.output<typeof decideSchema>;

const STORAGE_TTL = 60 * 60 * 12; // 12 hours

const storage = createStorage({
    driver: redisDriver({
        url: process.env.REDIS_URL!,
    }),
});

type Route = {
    name: string;
    description: string;
};

const routes = [
    {
        name: 'general_chat',
        description:
            "Everyday conversation, casual questions, opinions, and simple requests that don't require specialized reasoning, coding, or creative writing.",
    },
    {
        name: 'creative_writing',
        description:
            'Fiction, poetry, storytelling, or any request asking for imaginative or stylistically expressive writing, as opposed to factual or conversational text.',
    },
    {
        name: 'deep_reasoning',
        description:
            'Problems requiring careful multi-step logical, mathematical, or analytical reasoning, or that involve working through a large amount of context to answer accurately.',
    },
    {
        name: 'text_transformation',
        description:
            'Summarizing, translating, extracting structured data from, or reformatting existing text without adding new creative or analytical content.',
    },
    {
        name: 'vision_multimodal',
        description: 'Tasks that require interpreting an image, screenshot, diagram, or photo alongside text.',
    },
    {
        name: 'tool_use_agentic',
        description:
            'Non-coding tasks that require planning and executing a multi-step sequence of actions or external tool calls to reach a goal.',
    },
    {
        name: 'coding',
        description: 'Writing, reviewing, fixing, refactoring, or explaining code in a single-shot exchange.',
    },
    {
        name: 'agentic_coding',
        description:
            'Open-ended coding work requiring autonomous planning across multiple files or steps, running and interpreting tool output like tests or builds, and iterating until done.',
    },
] as const satisfies Route[];

type RouteName = (typeof routes)[number]['name'];

const returnedRouteSchema = z.object({
    route: z.array(z.string()),
});

type ReturnedRoute = z.output<typeof returnedRouteSchema>;

function routeToDecision(rte: RouteName | (string & {})): Decision | null {
    switch (rte) {
        // TODO other models? not sure.
        case 'text_transformation':
        case 'vision_multimodal':
        case 'agentic_coding':
        case 'coding':
        case 'tool_use_agentic':
        case 'general_chat': {
            return getDecisionForSharedModel('glm-5.3-flash');
        }
        case 'deep_reasoning':
        case 'creative_writing': {
            return getDecisionForSharedModel('glm-5.3');
        }
    }
    return null;
}

function routesToDecision(routes: ReturnedRoute): Decision {
    for (const route of routes.route) {
        const res = routeToDecision(route);
        if (res) {
            return res;
        }
    }
    return getDecisionForSharedModel('glm-5.3-flash');
}

const routesStr = JSON.stringify(routes, null, 2);

function buildOrchestratorPrompt(conversation: string) {
    return `You are a helpful assistant that selects the most suitable routes based on user intent.
You are provided with a list of available routes enclosed within <routes></routes> XML tags:
<routes>
${routesStr}
</routes>

You are also given the conversation context enclosed within <conversation></conversation> XML tags:
<conversation>
${conversation}
</conversation>

## Instructions
1. Analyze the latest user intent from the conversation.
2. Compare it against the available routes to find which routes can help fulfill the request.
3. Respond only with the exact route names from <routes>.
4. If no routes can help or the intent is already fulfilled, return an empty list.

## Response Format
Return your answer strictly in JSON as follows:
{"route": ["route_name_1", "route_name_2", "..."]}
If no routes are needed, return an empty list for \`route\`.`;
}

class ErrorWithResponse extends Error {
    constructor(readonly res: Response) {
        super('Error during request handling.');
    }
}

function auth(req: Request) {
    if (req.headers.get('authorization') !== `Bearer ${GATEWAY_TOKEN}`) {
        throw new ErrorWithResponse(new Response('unauthorized', { status: 401 }));
    }
}

type Method = 'POST' | 'GET' | 'DELETE' | 'HEAD' | 'PATCH' | 'PUT' | 'OPTIONS';

function method(req: Request, method: Method, ...methods: Method[]) {
    const all = [method, ...methods];
    for (const mtd of all) {
        if (req.method === mtd) {
            return;
        }
    }
    throw new ErrorWithResponse(
        new Response('Wrong method', {
            status: 405,
        }),
    );
}

async function json(req: Request) {
    try {
        return await req.json();
    } catch {
        throw new ErrorWithResponse(
            new Response('Invalid format', {
                status: 400,
            }),
        );
    }
}

async function schema<T extends z.ZodType>(req: Request, schema: T) {
    const j = await json(req);
    try {
        return schema.parse(j);
    } catch (err) {
        let message = 'Invalid input';
        if (err instanceof ZodError) {
            message = err.message;
        }
        throw new ErrorWithResponse(
            new Response(message, {
                status: 400,
            }),
        );
    }
}

async function getDecisionForReq(req: Request): Promise<Decision> {
    const model = req.headers.get('x-model-id');
    const provider = req.headers.get('x-provider-id');
    const session = req.headers.get('x-session-id');
    const parsed = decisionSchema.safeParse({ model, provider });
    if (parsed.success) {
        return parsed.data;
    }
    if (!session) {
        throw new ErrorWithResponse(
            new Response('Session or decision required!', {
                status: 400,
            }),
        );
    }
    const decision = await storage.get<Decision>(`route:${session}`);
    if (!decision) {
        throw new ErrorWithResponse(
            new Response('Session not found', {
                status: 404,
            }),
        );
    }
    return decision;
}

function validateVersion(req: BunRequest<`/:version/${string}`>): 'v2' | 'v3' | 'v4' {
    const { version } = req.params;
    switch (version) {
        case 'v2':
        case 'v3':
        case 'v4':
            return version;
    }
    throw new ErrorWithResponse(
        new Response('Invalid version', {
            status: 404,
        }),
    );
}

Bun.serve({
    port: 8080,
    routes: {
        '/healthz': async (req) => {
            auth(req);
            const sha = process.env.GIT_SHA;
            if (!sha) {
                return new Response('Not OK', {
                    status: 400,
                })
            }
            return new Response('OK', {
                status: 200,
                headers: {
                    'x-sam-ai-router': sha,
                },
            });
        },
        '/decide': async (req) => {
            auth(req);
            method(req, 'POST');
            const data = await schema(req, decideSchema);

            const sessionId = req.headers.get('x-session-id') ?? data.session ?? crypto.randomUUID();
            const storageKey = `route:${sessionId}`;

            let decision = await storage.get<Decision>(storageKey);
            if (decision) {
                return Response.json(decision, {
                    status: 200,
                    headers: {
                        'x-session-id': sessionId,
                    },
                });
            }
            const prompt = buildOrchestratorPrompt(data.conversation);
            const response = await generateText({
                model: routerModel,
                prompt,
            });
            let routes: ReturnedRoute = {
                route: [],
            };
            try {
                routes = returnedRouteSchema.parse(JSON.parse(response.text));
            } catch {}
            decision = routesToDecision(routes);
            await storage.set(storageKey, decision, {
                ttl: STORAGE_TTL,
            });
            return Response.json(decision, {
                status: 200,
                headers: {
                    'x-session-id': sessionId,
                },
            });
        },
        '/:version/generate': async (req) => {
            auth(req);
            method(req, 'POST');
            const version = validateVersion(req);
            const decision = await getDecisionForReq(req);
            const pr = COMPAT[version](provider('', { decision }));
            const j = await schema(req, getSchema(version));
            const genRes = await pr.doGenerate(j as any);
            return Response.json(genRes, {
                status: 200,
            });
        },
        '/:version/stream': async (req) => {
            auth(req);
            method(req, 'POST');
            const version = validateVersion(req);
            const decision = await getDecisionForReq(req);
            const pr = COMPAT[version](provider('', { decision }));
            const j = await schema(req, getSchema(version));
            const strRes = await pr.doStream(j as any);
            const enc = new TextEncoder();
            const tx = new TransformStream<
                LanguageModelV4StreamPart | LanguageModelV3StreamPart | LanguageModelV2StreamPart,
                Uint8Array
            >({
                start() {},
                transform(chunk, ctrl) {
                    ctrl.enqueue(enc.encode(JSON.stringify(chunk) + '\n'));
                },
            });
            const out = strRes.stream.pipeThrough(tx);
            return new Response(out, {
                status: 200,
                headers: {
                    'Content-Type': 'application/x-ndjson',
                },
            });
        },
    },
    error(err) {
        if (err instanceof ErrorWithResponse) {
            return err.res;
        }
        return new Response(err.message, {
            status: 500,
        });
    },
    async fetch(req) {
        auth(req);
        return new Response(null, {
            status: 404,
        });
    },
});
