import type {
    LanguageModelV3,
    LanguageModelV3CallOptions,
    LanguageModelV3GenerateResult,
    LanguageModelV3StreamPart,
    LanguageModelV3StreamResult,
    ProviderV3,
} from '@ai-sdk/provider';
import { createWorkersAI, type WorkersAI } from 'workers-ai-provider';
import {
    authHeaders,
    decide,
    DEFAULT_BASE_URL,
    parseNdjsonStream,
    type CustomChatSettings,
    type CustomLanguageOptions,
    type SamAIOptions,
} from './core';
import { COMPAT, v3CallOptionsToV4 } from './versions';

export type { SamAIOptions } from './core';

class CustomChatV3LanguageModel implements LanguageModelV3 {
    readonly specificationVersion = 'v3';
    readonly provider = 'sam-ai';

    private cfBindingProvider: WorkersAI | null = null;

    constructor(
        readonly modelId: string,
        private settings: CustomLanguageOptions,
    ) {
        if (settings.workersAI) {
            this.cfBindingProvider = createWorkersAI({ binding: settings.workersAI });
        }
    }
    readonly supportedUrls = {};

    async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
        const { decision, sessionId } = await decide(v3CallOptionsToV4(options), this.settings);
        if (decision.provider === 'cloudflare' && this.cfBindingProvider) {
            return COMPAT.v3(this.cfBindingProvider(decision.model)).doGenerate(options);
        }
        const response = await fetch(`${this.settings.baseUrl ?? DEFAULT_BASE_URL}/v3/generate`, {
            method: 'POST',
            body: JSON.stringify(options),
            headers: {
                ...authHeaders(this.settings),
                'x-session-id': sessionId,
                'x-model-id': decision.model,
                'x-provider-id': decision.provider,
            },
        });
        if (!response.ok) {
            throw new Error(await response.text());
        }
        const output = (await response.json()) as LanguageModelV3GenerateResult;

        output.providerMetadata ??= {};
        output.providerMetadata.session = {
            id: sessionId,
        };
        return output;
    }

    async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
        const { decision, sessionId } = await decide(v3CallOptionsToV4(options), this.settings);
        if (decision.provider === 'cloudflare' && this.cfBindingProvider) {
            return COMPAT.v3(this.cfBindingProvider(decision.model)).doStream(options);
        }
        const response = await fetch(`${this.settings.baseUrl ?? DEFAULT_BASE_URL}/v3/stream`, {
            method: 'POST',
            body: JSON.stringify(options),
            headers: {
                ...authHeaders(this.settings),
                'x-session-id': sessionId,
                'x-model-id': decision.model,
                'x-provider-id': decision.provider,
            },
        });
        if (!response.ok) {
            throw new Error(await response.text());
        }
        const stream = response.body;
        if (!stream) {
            throw new Error('No response body');
        }
        return {
            stream: parseNdjsonStream<LanguageModelV3StreamPart>(stream),
        };
    }
}

interface MainProviderV3 extends ProviderV3 {
    (modelId: string, settings?: CustomChatSettings): CustomChatV3LanguageModel;

    languageModel(modelId: string, settings?: CustomChatSettings): CustomChatV3LanguageModel;
}

export function createSamAIV3(options: SamAIOptions) {
    function languageModel(modelId: string, settings?: CustomChatSettings) {
        return new CustomChatV3LanguageModel(modelId, {
            ...settings,
            ...options,
        });
    }

    const provider = function (modelId: string, settings?: CustomChatSettings) {
        if (new.target) {
            throw new Error('The model factory function cannot be called with the new keyword.');
        }

        return languageModel(modelId, settings);
    };

    provider.languageModel = languageModel;

    return provider as MainProviderV3;
}
