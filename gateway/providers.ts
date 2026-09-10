import { createAiGateway } from 'ai-gateway-provider';
import { createUnified } from 'ai-gateway-provider/providers/unified';
import { createDeepSeek } from '@ai-sdk/deepseek';
import { createMoonshotAI } from '@ai-sdk/moonshotai';
import { createZai } from '@ai-sdk/zai';
import { z } from 'zod';
import {
    type ProviderV4,
    type LanguageModelV4,
    type LanguageModelV4CallOptions,
    type LanguageModelV4GenerateResult,
    type LanguageModelV4StreamResult,
} from '@ai-sdk/provider';
import type { Decision, Provider } from './types';

const accountId = process.env.CLOUDFLARE_ACCOUNT_ID!;
const apiKey = process.env.CLOUDFLARE_AI_GATEWAY_API_KEY!;
const aigateway = createAiGateway({
    accountId,
    gateway: 'default',
    apiKey,
});

const unified = createUnified();

function cloudflare(model: string) {
    return aigateway(unified(model));
}

function makeDS() {
    if (!process.env.DEEPSEEK_API_KEY) {
        return null;
    }
    return createDeepSeek({
        apiKey: process.env.DEEPSEEK_API_KEY!,
    });
}

function makeMS() {
    if (!process.env.MOONSHOT_API_KEY) {
        return null;
    }
    return createMoonshotAI({
        apiKey: process.env.MOONSHOT_API_KEY!,
    });
}

function makeZH() {
    if (!process.env.ZAI_API_KEY) {
        return null;
    }
    return createZai({
        apiKey: process.env.ZAI_API_KEY!,
    });
}

const PROVIDERS = {
    cloudflare,
    deepseek: makeDS(),
    moonshot: makeMS(),
    zhipu: makeZH(),
};

function makeModels<TDefault extends string, const TRest extends ReadonlyArray<string>>(
    def: TDefault,
    ...rest: TRest
): {
    models: [TDefault, ...TRest];
    default: TDefault;
} {
    return {
        models: [def, ...rest],
        default: def,
    };
}

export const MODELS = {
    cloudflare: makeModels(
        '@cf/zai-org/glm-5.3-flash',
        '@cf/zai-org/glm-5.3',
        '@cf/deepseek-ai/deepseek-v4-pro-0813',
        '@cf/deepseek-ai/deepseek-v4-flash-0731',
        '@cf/qwen/qwen3.8-27b',
        '@cf/zai-org/glm-5.2',
        '@cf/moonshotai/kimi-k2.7-code',
        '@cf/moonshotai/kimi-k2.6',
        '@cf/google/gemma-4-26b-a4b-it',
        '@cf/zai-org/glm-4.7-flash',
        '@cf/openai/gpt-oss-20b',
        '@cf/openai/gpt-oss-120b',
        '@cf/qwen/qwen3-30b-a3b-fp8',
    ),
    deepseek: makeModels(
        'deepseek-flash',
        // 'deepseek-pro', // Inferred given deepseek-flash = deepseek-v4.1-flash
        'deepseek-v4-flash',
        'deepseek-v4-pro',
        'deepseek-v4-flash-vision-exp',
    ),
    moonshot: makeModels('kimi-k2.7-code-highspeed', 'kimi-k3', 'kimi-k2.7-code', 'kimi-k2.6'),
    zhipu: makeModels('glm-5.3-flash', 'glm-5.3', 'glm-5.2'),
} as const satisfies Record<
    keyof typeof PROVIDERS,
    {
        models: string[];
        default: string;
    }
>;

type SharedModel =
    | 'glm-5.3-flash'
    | 'glm-5.3'
    | 'glm-5.2'
    | 'kimi-k2.7-code'
    | 'kimi-k2.6'
    | 'deepseek-v4-flash'
    | 'deepseek-v4-pro';

function toProviderModel(shared: SharedModel, provider: Provider): string {
    switch (provider) {
        case 'deepseek':
        case 'moonshot':
        case 'zhipu': {
            return shared;
        }
    }
    switch (shared) {
        case 'deepseek-v4-flash':
            return '@cf/deepseek-ai/deepseek-v4-flash-0731';
        case 'deepseek-v4-pro':
            return '@cf/deepseek-ai/deepseek-v4-pro-0813';
    }
    if (shared.startsWith('glm')) {
        return `@cf/zai-org/${shared}`;
    }
    if (shared.startsWith('kimi')) {
        return `@cf/moonshotai/${shared}`;
    }
    if (shared.startsWith('deepseek')) {
        return `@cf/deepseek-ai/${shared}`;
    }
    return MODELS.cloudflare.default;
}

function getProviderForSharedModel(model: SharedModel): Provider {
    // TODO budget checks
    switch (model) {
        case 'glm-5.3':
        case 'glm-5.2':
        case 'glm-5.3-flash': {
            if (PROVIDERS.zhipu) {
                return 'zhipu';
            }
            return 'cloudflare';
        }
        case 'deepseek-v4-flash':
        case 'deepseek-v4-pro': {
            if (PROVIDERS.deepseek) {
                return 'deepseek';
            }
            return 'cloudflare';
        }
        case 'kimi-k2.6':
        case 'kimi-k2.7-code': {
            if (PROVIDERS.moonshot) {
                return 'moonshot';
            }
            return 'cloudflare';
        }
    }
    return 'cloudflare';
}

export function getDecisionForSharedModel(model: SharedModel): Decision {
    const provider = getProviderForSharedModel(model);
    const realModel = toProviderModel(model, provider);
    return {
        provider,
        model: realModel,
    };
}

function modelWithFallback(provider: keyof typeof MODELS, ...models: string[]) {
    const mds = MODELS[provider];
    for (const model of models) {
        if (mds.models.includes(model as never)) {
            return model;
        }
    }
    return mds.default;
}

interface CustomChatSettings {
    decision: Decision;
}

class CustomChatLanguageModel implements LanguageModelV4 {
    readonly specificationVersion = 'v4';
    readonly provider: Provider;
    readonly modelId: string;

    private internalProvider: LanguageModelV4;

    constructor(
        modelId: string,
        readonly settings: CustomChatSettings,
    ) {
        let mp = PROVIDERS[settings.decision.provider];
        if (mp) {
            this.provider = settings.decision.provider;
        } else {
            this.provider = 'cloudflare';
            mp = PROVIDERS.cloudflare;
        }
        const md = modelWithFallback(this.provider, settings.decision.model, modelId);
        this.modelId = md;
        this.internalProvider = mp(md);
    }
    readonly supportedUrls = {};

    doGenerate(options: LanguageModelV4CallOptions): PromiseLike<LanguageModelV4GenerateResult> {
        return this.internalProvider.doGenerate(options);
    }
    doStream(options: LanguageModelV4CallOptions): PromiseLike<LanguageModelV4StreamResult> {
        return this.internalProvider.doStream(options);
    }
}

interface MainProvider extends ProviderV4 {
    (modelId: string, settings?: CustomChatSettings): CustomChatLanguageModel;

    // Add specific methods for different model types
    languageModel(modelId: string, settings?: CustomChatSettings): CustomChatLanguageModel;
}

function mk() {
    function languageModel(modelId: string, settings?: CustomChatSettings) {
        if (!settings) {
            return cloudflare(MODELS.cloudflare.default);
        }
        return new CustomChatLanguageModel(modelId, settings);
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

export const provider = mk();
