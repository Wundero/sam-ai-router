import { loadApiKey } from '@ai-sdk/provider-utils';
import type { LanguageModelV4CallOptions, LanguageModelV4Prompt } from '@ai-sdk/provider';
import type { WorkersAISettings } from 'workers-ai-provider';
import { z } from 'zod';
import { decisionSchema, type Decision } from './types';

export const DEFAULT_BASE_URL = 'https://ai.1d.gg';

/** A Cloudflare Workers AI binding (`env.AI`), as accepted by `workers-ai-provider`. */
export type SamAIWorkersBinding = NonNullable<WorkersAISettings['binding']>;

/** Minimal storage interface. An `unstorage` `Storage` instance satisfies it. */
export interface SamAIStorage {
    get<T>(key: string): Promise<T | null>;
}

export interface SamAIOptions {
    apiKey?: string;
    baseUrl?: string;
    storage?: SamAIStorage;
    workersAI?: SamAIWorkersBinding;
}

export interface CustomLanguageOptions extends SamAIOptions {
    decision?: Decision;
    sessionId?: string;
}

export interface CustomChatSettings {
    decision: Decision;
}

const sessionIdSchema = z.string();

export function prepareConversationForDecision(prompt: LanguageModelV4Prompt) {
    return prompt.map((message) => {
        switch (message.role) {
            case 'system':
                return { role: 'system', content: message.content };

            case 'user':
                return {
                    role: 'user',
                    content: message.content
                        .map((part) => {
                            switch (part.type) {
                                case 'text':
                                    return { type: 'text', text: part.text };
                                case 'file':
                                    return {
                                        type: 'file',
                                        mime: part.mediaType,
                                        name: part.filename,
                                    };
                                default:
                                    return null;
                            }
                        })
                        .filter(Boolean),
                };

            case 'assistant':
                return {
                    role: message.role,
                    content: message.content
                        .map((item) => {
                            switch (item.type) {
                                case 'custom': {
                                    return {
                                        type: item.type,
                                        kind: item.kind,
                                    };
                                }
                                case 'file': {
                                    return {
                                        type: 'file',
                                        mime: item.mediaType,
                                        name: item.filename,
                                    };
                                }
                                case 'text': {
                                    return {
                                        type: item.type,
                                        text: item.text,
                                    };
                                }
                            }
                            return null;
                        })
                        .filter(Boolean),
                };

            default:
                return null;
        }
    });
}

export function authHeaders(settings: SamAIOptions): Record<string, string> {
    return {
        Authorization: `Bearer ${loadApiKey({
            apiKey: settings.apiKey,
            description: 'Sam AI Gateway API key',
            environmentVariableName: 'SAM_AI_API_KEY',
        })}`,
    };
}

export async function decide(
    options: LanguageModelV4CallOptions,
    settings: CustomLanguageOptions,
): Promise<{ decision: Decision; sessionId: string }> {
    const baseUrl = settings.baseUrl ?? DEFAULT_BASE_URL;
    const paramSessionId = sessionIdSchema.safeParse(options.providerOptions?.session?.id);
    const settingsSessionId = settings.sessionId;
    const sessionId = paramSessionId.data ?? settingsSessionId;
    if (sessionId) {
        const stored = await settings.storage?.get<Decision>(`$sam-ai::session-decision::${sessionId}`);
        if (stored) {
            return {
                decision: stored,
                sessionId,
            };
        }
    }
    const prepped = prepareConversationForDecision(options.prompt);
    const decision = await fetch(`${baseUrl}/decide`, {
        method: 'POST',
        headers: {
            ...authHeaders(settings),
            ...(sessionId
                ? {
                      'x-session-id': sessionId,
                  }
                : {}),
        },
        body: JSON.stringify({
            session: sessionId,
            conversation: JSON.stringify(prepped),
        }),
    });
    if (!decision.ok) {
        throw new Error(await decision.text());
    }
    const realSessionId = decision.headers.get('x-session-id') ?? sessionId;
    if (!realSessionId) {
        throw new Error('Invalid state!');
    }
    return {
        decision: decisionSchema.parse(await decision.json()),
        sessionId: realSessionId,
    };
}

export function parseNdjsonStream<T>(stream: ReadableStream<Uint8Array>): ReadableStream<T> {
    const dec = new TextDecoder();
    let buf = '';
    return stream.pipeThrough(
        new TransformStream<Uint8Array, T>({
            start() {},
            transform(chunk, ctrl) {
                const str = dec.decode(chunk, {
                    stream: true,
                });
                buf += str;
                const lines = buf.split('\n');
                buf = lines.pop()!;
                for (const line of lines) {
                    const part = JSON.parse(line) as T;
                    ctrl.enqueue(part);
                }
            },
            flush(ctrl) {
                if (buf.trim()) {
                    ctrl.error(new Error(`Stream ended mid-message: ${buf.slice(0, 100)}`));
                }
            },
        }),
    );
}
