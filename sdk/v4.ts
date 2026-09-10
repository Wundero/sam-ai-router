import type {
    LanguageModelV4,
    LanguageModelV4CallOptions,
    LanguageModelV4GenerateResult,
    LanguageModelV4StreamPart,
    LanguageModelV4StreamResult,
    ProviderV4,
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

export type { SamAIOptions } from './core';

class CustomChatLanguageModel implements LanguageModelV4 {
    readonly specificationVersion = 'v4';
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

    async doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
        const { decision, sessionId } = await decide(options, this.settings);
        if (decision.provider === 'cloudflare' && this.cfBindingProvider) {
            return this.cfBindingProvider(decision.model).doGenerate(options);
        }
        const response = await fetch(`${this.settings.baseUrl ?? DEFAULT_BASE_URL}/v4/generate`, {
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
        const output = (await response.json()) as LanguageModelV4GenerateResult;

        output.providerMetadata ??= {};
        output.providerMetadata.session = {
            id: sessionId,
        };
        return output;
    }

    async doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
        const { decision, sessionId } = await decide(options, this.settings);
        if (decision.provider === 'cloudflare' && this.cfBindingProvider) {
            return this.cfBindingProvider(decision.model).doStream(options);
        }
        const response = await fetch(`${this.settings.baseUrl ?? DEFAULT_BASE_URL}/v4/stream`, {
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
            stream: parseNdjsonStream<LanguageModelV4StreamPart>(stream),
        };
    }
}

interface MainProvider extends ProviderV4 {
    (modelId: string, settings?: CustomChatSettings): CustomChatLanguageModel;

    languageModel(modelId: string, settings?: CustomChatSettings): CustomChatLanguageModel;
}

export function createSamAI(options: SamAIOptions) {
    function languageModel(modelId: string, settings?: CustomChatSettings) {
        return new CustomChatLanguageModel(modelId, {
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

    return provider as MainProvider;
}
