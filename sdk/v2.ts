import type {
    LanguageModelV2,
    LanguageModelV2CallOptions,
    LanguageModelV2StreamPart,
    ProviderV2,
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
import { COMPAT, v2CallOptionsToV4 } from './versions';

export type { SamAIOptions } from './core';

type V2GenerateResult = Awaited<ReturnType<LanguageModelV2['doGenerate']>>;
type V2StreamResult = Awaited<ReturnType<LanguageModelV2['doStream']>>;

class CustomChatV2LanguageModel implements LanguageModelV2 {
    readonly specificationVersion = 'v2';
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

    async doGenerate(options: LanguageModelV2CallOptions): Promise<V2GenerateResult> {
        const { decision, sessionId } = await decide(v2CallOptionsToV4(options), this.settings);
        if (decision.provider === 'cloudflare' && this.cfBindingProvider) {
            return COMPAT.v2(this.cfBindingProvider(decision.model)).doGenerate(options);
        }
        const response = await fetch(`${this.settings.baseUrl ?? DEFAULT_BASE_URL}/v2/generate`, {
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
        const output = (await response.json()) as V2GenerateResult;

        output.providerMetadata ??= {};
        output.providerMetadata.session = {
            id: sessionId,
        };
        return output;
    }

    async doStream(options: LanguageModelV2CallOptions): Promise<V2StreamResult> {
        const { decision, sessionId } = await decide(v2CallOptionsToV4(options), this.settings);
        if (decision.provider === 'cloudflare' && this.cfBindingProvider) {
            return COMPAT.v2(this.cfBindingProvider(decision.model)).doStream(options);
        }
        const response = await fetch(`${this.settings.baseUrl ?? DEFAULT_BASE_URL}/v2/stream`, {
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
            stream: parseNdjsonStream<LanguageModelV2StreamPart>(stream),
        };
    }
}

interface MainProviderV2 extends ProviderV2 {
    (modelId: string, settings?: CustomChatSettings): CustomChatV2LanguageModel;

    languageModel(modelId: string, settings?: CustomChatSettings): CustomChatV2LanguageModel;
}

export function createSamAIV2(options: SamAIOptions) {
    function languageModel(modelId: string, settings?: CustomChatSettings) {
        return new CustomChatV2LanguageModel(modelId, {
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

    return provider as MainProviderV2;
}
