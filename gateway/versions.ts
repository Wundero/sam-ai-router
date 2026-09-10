import type {
    LanguageModelV2,
    LanguageModelV2CallOptions,
    LanguageModelV2CallWarning,
    LanguageModelV2Content,
    LanguageModelV2File,
    LanguageModelV2FilePart,
    LanguageModelV2FunctionTool,
    LanguageModelV2Message,
    LanguageModelV2ProviderDefinedTool,
    LanguageModelV2ReasoningPart,
    LanguageModelV2Source,
    LanguageModelV2StreamPart,
    LanguageModelV2TextPart,
    LanguageModelV2ToolCall,
    LanguageModelV2ToolCallPart,
    LanguageModelV2ToolResultOutput,
    LanguageModelV2ToolResultPart,
    LanguageModelV2Usage,
    LanguageModelV3,
    LanguageModelV3CallOptions,
    LanguageModelV3Content,
    LanguageModelV3File,
    LanguageModelV3FilePart,
    LanguageModelV3GenerateResult,
    LanguageModelV3Message,
    LanguageModelV3ReasoningPart,
    LanguageModelV3StreamPart,
    LanguageModelV3StreamResult,
    LanguageModelV3TextPart,
    LanguageModelV3ToolApprovalResponsePart,
    LanguageModelV3ToolCallPart,
    LanguageModelV3ToolResultOutput,
    LanguageModelV3ToolResultPart,
    LanguageModelV4,
    LanguageModelV4CallOptions,
    LanguageModelV4Content,
    LanguageModelV4CustomPart,
    LanguageModelV4File,
    LanguageModelV4FilePart,
    LanguageModelV4FunctionTool,
    LanguageModelV4Message,
    LanguageModelV4ProviderTool,
    LanguageModelV4ReasoningFilePart,
    LanguageModelV4ReasoningPart,
    LanguageModelV4Source,
    LanguageModelV4StreamPart,
    LanguageModelV4TextPart,
    LanguageModelV4ToolApprovalResponsePart,
    LanguageModelV4ToolCallPart,
    LanguageModelV4ToolCall,
    LanguageModelV4ToolResult,
    LanguageModelV4ToolResultOutput,
    LanguageModelV4ToolResultPart,
    LanguageModelV4Usage,
    SharedV2ProviderMetadata,
    SharedV3Warning,
    SharedV4FileData,
    SharedV4ProviderMetadata,
    SharedV4ProviderReference,
    SharedV4Warning,
} from '@ai-sdk/provider';

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

type V2PromptPart =
    | LanguageModelV2TextPart
    | LanguageModelV2FilePart
    | LanguageModelV2ReasoningPart
    | LanguageModelV2ToolCallPart
    | LanguageModelV2ToolResultPart;

type V3PromptPart =
    | LanguageModelV3TextPart
    | LanguageModelV3FilePart
    | LanguageModelV3ReasoningPart
    | LanguageModelV3ToolCallPart
    | LanguageModelV3ToolResultPart
    | LanguageModelV3ToolApprovalResponsePart;

type V4PromptPart =
    | LanguageModelV4TextPart
    | LanguageModelV4FilePart
    | LanguageModelV4CustomPart
    | LanguageModelV4ReasoningPart
    | LanguageModelV4ReasoningFilePart
    | LanguageModelV4ToolCallPart
    | LanguageModelV4ToolResultPart
    | LanguageModelV4ToolApprovalResponsePart;

type V2ToolResult = Extract<LanguageModelV2Content, { type: 'tool-result' }>;

type V4ToolResultContentItem =
    LanguageModelV4ToolResultOutput extends infer O
        ? O extends { type: 'content'; value: Array<infer C> }
            ? C
            : never
        : never;

type V3ToolResultContentItem =
    LanguageModelV3ToolResultOutput extends infer O
        ? O extends { type: 'content'; value: Array<infer C> }
            ? C
            : never
        : never;

type V2GenerateResult = Awaited<ReturnType<LanguageModelV2['doGenerate']>>;
type V2StreamResult = Awaited<ReturnType<LanguageModelV2['doStream']>>;

function notNull<T>(value: T | null): value is T {
    return value !== null;
}

// V2 metadata nests JSONValue records; V3/V4 use JSONObject. JSONObject's
// `undefined` values never survive serialization, so the downcast is safe.
function toV2Meta(meta: SharedV4ProviderMetadata | undefined): SharedV2ProviderMetadata | undefined {
    return meta as SharedV2ProviderMetadata | undefined;
}

function toV4FileData(data: string | Uint8Array | URL): SharedV4FileData {
    if (data instanceof URL) {
        return { type: 'url', url: data };
    }
    return { type: 'data', data };
}

function fromV4FileData(data: SharedV4FileData): string | Uint8Array {
    switch (data.type) {
        case 'data':
            return data.data;
        case 'url':
            return data.url.toString();
        case 'text':
            return data.text;
        case 'reference':
            return JSON.stringify(data.reference);
    }
}

function toProviderReference(fileId: string | Record<string, string>): SharedV4ProviderReference {
    return (typeof fileId === 'string' ? { default: fileId } : fileId) as SharedV4ProviderReference;
}

function mapStream<T, U>(stream: ReadableStream<T>, map: (chunk: T) => U | null): ReadableStream<U> {
    const reader = stream.getReader();
    return new ReadableStream<U>({
        async pull(controller) {
            const { done, value } = await reader.read();
            if (done) {
                controller.close();
                return;
            }
            const mapped = map(value);
            if (mapped !== null) {
                controller.enqueue(mapped);
            }
        },
        async cancel(reason) {
            await reader.cancel(reason);
        },
    });
}

// ---------------------------------------------------------------------------
// V2 -> V4 (request)
// ---------------------------------------------------------------------------

function v2ToolResultOutputToV4(output: LanguageModelV2ToolResultOutput): LanguageModelV4ToolResultOutput {
    if (output.type !== 'content') {
        return output;
    }
    return {
        type: 'content',
        value: output.value.map((item) =>
            item.type === 'media'
                ? { type: 'file', data: { type: 'data', data: item.data }, mediaType: item.mediaType }
                : item,
        ),
    };
}

function v2PartToV4(part: V2PromptPart): V4PromptPart {
    switch (part.type) {
        case 'file':
            return {
                type: 'file',
                filename: part.filename,
                data: toV4FileData(part.data),
                mediaType: part.mediaType,
                providerOptions: part.providerOptions,
            };
        case 'tool-result':
            return {
                type: 'tool-result',
                toolCallId: part.toolCallId,
                toolName: part.toolName,
                output: v2ToolResultOutputToV4(part.output),
                providerOptions: part.providerOptions,
            };
        default:
            return part;
    }
}

function v2MessageToV4(message: LanguageModelV2Message): LanguageModelV4Message {
    switch (message.role) {
        case 'system':
            return { role: 'system', content: message.content, providerOptions: message.providerOptions };
        case 'user':
            return {
                role: 'user',
                content: message.content.map(v2PartToV4) as Array<LanguageModelV4TextPart | LanguageModelV4FilePart>,
                providerOptions: message.providerOptions,
            };
        case 'assistant':
            return {
                role: 'assistant',
                content: message.content.map(v2PartToV4) as Array<
                    | LanguageModelV4TextPart
                    | LanguageModelV4FilePart
                    | LanguageModelV4ReasoningPart
                    | LanguageModelV4ToolCallPart
                    | LanguageModelV4ToolResultPart
                >,
                providerOptions: message.providerOptions,
            };
        case 'tool':
            return {
                role: 'tool',
                content: message.content.map(v2PartToV4) as Array<LanguageModelV4ToolResultPart>,
                providerOptions: message.providerOptions,
            };
    }
}

function v2ToolToV4(
    tool: LanguageModelV2FunctionTool | LanguageModelV2ProviderDefinedTool,
): LanguageModelV4FunctionTool | LanguageModelV4ProviderTool {
    if (tool.type === 'provider-defined') {
        return { type: 'provider', id: tool.id, name: tool.name, args: tool.args };
    }
    return tool;
}

function v2CallOptionsToV4(options: LanguageModelV2CallOptions): LanguageModelV4CallOptions {
    return {
        ...options,
        prompt: options.prompt.map(v2MessageToV4),
        tools: options.tools?.map(v2ToolToV4),
    };
}

// ---------------------------------------------------------------------------
// V3 -> V4 (request)
// ---------------------------------------------------------------------------

function v3ToolResultContentToV4(item: V3ToolResultContentItem): V4ToolResultContentItem {
    switch (item.type) {
        case 'file-data':
            return {
                type: 'file',
                data: { type: 'data', data: item.data },
                mediaType: item.mediaType,
                filename: item.filename,
                providerOptions: item.providerOptions,
            };
        case 'file-url':
            return { type: 'file', data: { type: 'url', url: new URL(item.url) }, mediaType: '', providerOptions: item.providerOptions };
        case 'file-id':
            return {
                type: 'file',
                data: { type: 'reference', reference: toProviderReference(item.fileId) },
                mediaType: '',
                providerOptions: item.providerOptions,
            };
        case 'image-data':
            return { type: 'file', data: { type: 'data', data: item.data }, mediaType: item.mediaType, providerOptions: item.providerOptions };
        case 'image-url':
            return { type: 'file', data: { type: 'url', url: new URL(item.url) }, mediaType: '', providerOptions: item.providerOptions };
        case 'image-file-id':
            return {
                type: 'file',
                data: { type: 'reference', reference: toProviderReference(item.fileId) },
                mediaType: '',
                providerOptions: item.providerOptions,
            };
        case 'text':
        case 'custom':
            return item;
    }
}

function v3ToolResultOutputToV4(output: LanguageModelV3ToolResultOutput): LanguageModelV4ToolResultOutput {
    if (output.type !== 'content') {
        return output;
    }
    return { type: 'content', value: output.value.map(v3ToolResultContentToV4) };
}

function v3PartToV4(part: V3PromptPart): V4PromptPart {
    switch (part.type) {
        case 'file':
            return {
                type: 'file',
                filename: part.filename,
                data: toV4FileData(part.data),
                mediaType: part.mediaType,
                providerOptions: part.providerOptions,
            };
        case 'tool-result':
            return {
                type: 'tool-result',
                toolCallId: part.toolCallId,
                toolName: part.toolName,
                output: v3ToolResultOutputToV4(part.output),
                providerOptions: part.providerOptions,
            };
        default:
            return part;
    }
}

function v3MessageToV4(message: LanguageModelV3Message): LanguageModelV4Message {
    switch (message.role) {
        case 'system':
            return { role: 'system', content: message.content, providerOptions: message.providerOptions };
        case 'user':
            return {
                role: 'user',
                content: message.content.map(v3PartToV4) as Array<LanguageModelV4TextPart | LanguageModelV4FilePart>,
                providerOptions: message.providerOptions,
            };
        case 'assistant':
            return {
                role: 'assistant',
                content: message.content.map(v3PartToV4) as Array<
                    | LanguageModelV4TextPart
                    | LanguageModelV4FilePart
                    | LanguageModelV4ReasoningPart
                    | LanguageModelV4ToolCallPart
                    | LanguageModelV4ToolResultPart
                >,
                providerOptions: message.providerOptions,
            };
        case 'tool':
            return {
                role: 'tool',
                content: message.content.map(v3PartToV4) as Array<
                    LanguageModelV4ToolResultPart | LanguageModelV4ToolApprovalResponsePart
                >,
                providerOptions: message.providerOptions,
            };
    }
}

function v3CallOptionsToV4(options: LanguageModelV3CallOptions): LanguageModelV4CallOptions {
    return {
        ...options,
        prompt: options.prompt.map(v3MessageToV4),
    };
}

// ---------------------------------------------------------------------------
// V4 -> V2 (response)
// ---------------------------------------------------------------------------

function v4FileToV2(file: LanguageModelV4File): LanguageModelV2File {
    return { type: 'file', mediaType: file.mediaType, data: fromV4FileData(file.data) };
}

function v4ToolCallToV2(call: LanguageModelV4ToolCall): LanguageModelV2ToolCall {
    return {
        type: 'tool-call',
        toolCallId: call.toolCallId,
        toolName: call.toolName,
        input: call.input,
        providerExecuted: call.providerExecuted,
        providerMetadata: toV2Meta(call.providerMetadata),
    };
}

function v4ToolResultToV2(result: LanguageModelV4ToolResult): V2ToolResult {
    return {
        type: 'tool-result',
        toolCallId: result.toolCallId,
        toolName: result.toolName,
        result: result.result,
        isError: result.isError,
        providerMetadata: toV2Meta(result.providerMetadata),
    };
}

function v4SourceToV2(source: LanguageModelV4Source): LanguageModelV2Source {
    return { ...source, providerMetadata: toV2Meta(source.providerMetadata) };
}

function v4UsageToV2(usage: LanguageModelV4Usage): LanguageModelV2Usage {
    const input = usage.inputTokens.total;
    const output = usage.outputTokens.total;
    return {
        inputTokens: input,
        outputTokens: output,
        totalTokens: input !== undefined || output !== undefined ? (input ?? 0) + (output ?? 0) : undefined,
        reasoningTokens: usage.outputTokens.reasoning,
        cachedInputTokens: usage.inputTokens.cacheRead,
    };
}

function v4WarningsToV2(warnings: Array<SharedV4Warning>): Array<LanguageModelV2CallWarning> {
    return warnings.map((warning) => {
        switch (warning.type) {
            case 'unsupported':
                return {
                    type: 'other',
                    message: `Unsupported feature: ${warning.feature}${warning.details ? ` (${warning.details})` : ''}`,
                };
            case 'compatibility':
                return {
                    type: 'other',
                    message: `Compatibility feature: ${warning.feature}${warning.details ? ` (${warning.details})` : ''}`,
                };
            case 'deprecated':
                return { type: 'other', message: `Deprecated: ${warning.setting}. ${warning.message}` };
            case 'other':
                return { type: 'other', message: warning.message };
        }
    });
}

function v4ContentToV2(content: LanguageModelV4Content): LanguageModelV2Content | null {
    switch (content.type) {
        case 'text':
            return { type: 'text', text: content.text, providerMetadata: toV2Meta(content.providerMetadata) };
        case 'reasoning':
            return { type: 'reasoning', text: content.text, providerMetadata: toV2Meta(content.providerMetadata) };
        case 'file':
            return v4FileToV2(content);
        case 'source':
            return v4SourceToV2(content);
        case 'tool-call':
            return v4ToolCallToV2(content);
        case 'tool-result':
            return v4ToolResultToV2(content);
        case 'custom':
        case 'reasoning-file':
        case 'tool-approval-request':
            return null;
    }
}

function v4StreamPartToV2(part: LanguageModelV4StreamPart): LanguageModelV2StreamPart | null {
    switch (part.type) {
        case 'text-start':
        case 'text-delta':
        case 'text-end':
        case 'reasoning-start':
        case 'reasoning-delta':
        case 'reasoning-end':
        case 'tool-input-delta':
        case 'tool-input-end':
            return { ...part, providerMetadata: toV2Meta(part.providerMetadata) };
        case 'tool-input-start':
            return {
                type: 'tool-input-start',
                id: part.id,
                toolName: part.toolName,
                providerMetadata: toV2Meta(part.providerMetadata),
                providerExecuted: part.providerExecuted,
            };
        case 'tool-call':
            return v4ToolCallToV2(part);
        case 'tool-result':
            return v4ToolResultToV2(part);
        case 'file':
            return v4FileToV2(part);
        case 'source':
            return v4SourceToV2(part);
        case 'stream-start':
            return { type: 'stream-start', warnings: v4WarningsToV2(part.warnings) };
        case 'finish':
            return {
                type: 'finish',
                usage: v4UsageToV2(part.usage),
                finishReason: part.finishReason.unified,
                providerMetadata: toV2Meta(part.providerMetadata),
            };
        case 'response-metadata':
        case 'raw':
        case 'error':
            return part;
        case 'custom':
        case 'reasoning-file':
        case 'tool-approval-request':
            return null;
    }
}

// ---------------------------------------------------------------------------
// V4 -> V3 (response)
// ---------------------------------------------------------------------------

function v4FileToV3(file: LanguageModelV4File): LanguageModelV3File {
    return { type: 'file', mediaType: file.mediaType, data: fromV4FileData(file.data), providerMetadata: file.providerMetadata };
}

function v4WarningsToV3(warnings: Array<SharedV4Warning>): Array<SharedV3Warning> {
    const out: Array<SharedV3Warning> = [];
    for (const warning of warnings) {
        switch (warning.type) {
            case 'unsupported':
                out.push({ type: 'unsupported', feature: warning.feature, details: warning.details });
                break;
            case 'compatibility':
                out.push({ type: 'compatibility', feature: warning.feature, details: warning.details });
                break;
            case 'deprecated':
                out.push({ type: 'other', message: `Deprecated: ${warning.setting}. ${warning.message}` });
                break;
            case 'other':
                out.push({ type: 'other', message: warning.message });
                break;
        }
    }
    return out;
}

function v4ContentToV3(content: LanguageModelV4Content): LanguageModelV3Content | null {
    switch (content.type) {
        case 'file':
            return v4FileToV3(content);
        case 'custom':
        case 'reasoning-file':
            return null;
        default:
            return content;
    }
}

function v4StreamPartToV3(part: LanguageModelV4StreamPart): LanguageModelV3StreamPart | null {
    switch (part.type) {
        case 'file':
            return v4FileToV3(part);
        case 'stream-start':
            return { type: 'stream-start', warnings: v4WarningsToV3(part.warnings) };
        case 'custom':
        case 'reasoning-file':
            return null;
        default:
            return part;
    }
}

// ---------------------------------------------------------------------------
// Adapters
// ---------------------------------------------------------------------------

class V2Adapter implements LanguageModelV2 {
    readonly specificationVersion = 'v2';
    provider: string;
    modelId: string;
    supportedUrls: PromiseLike<Record<string, RegExp[]>> | Record<string, RegExp[]>;

    constructor(private src: LanguageModelV4) {
        this.provider = src.provider;
        this.modelId = src.modelId;
        this.supportedUrls = src.supportedUrls;
    }

    async doGenerate(options: LanguageModelV2CallOptions): Promise<V2GenerateResult> {
        const result = await this.src.doGenerate(v2CallOptionsToV4(options));
        return {
            content: result.content.map(v4ContentToV2).filter(notNull),
            finishReason: result.finishReason.unified,
            usage: v4UsageToV2(result.usage),
            warnings: v4WarningsToV2(result.warnings),
            providerMetadata: toV2Meta(result.providerMetadata),
            request: result.request,
            response: result.response,
        };
    }

    async doStream(options: LanguageModelV2CallOptions): Promise<V2StreamResult> {
        const result = await this.src.doStream(v2CallOptionsToV4(options));
        return {
            stream: mapStream(result.stream, v4StreamPartToV2),
            request: result.request,
            response: result.response,
        };
    }
}

class V3Adapter implements LanguageModelV3 {
    readonly specificationVersion = 'v3';
    provider: string;
    modelId: string;
    supportedUrls: PromiseLike<Record<string, RegExp[]>> | Record<string, RegExp[]>;

    constructor(private src: LanguageModelV4) {
        this.provider = src.provider;
        this.modelId = src.modelId;
        this.supportedUrls = src.supportedUrls;
    }

    async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
        const result = await this.src.doGenerate(v3CallOptionsToV4(options));
        return {
            content: result.content.map(v4ContentToV3).filter(notNull),
            finishReason: result.finishReason,
            usage: result.usage,
            warnings: v4WarningsToV3(result.warnings),
            providerMetadata: result.providerMetadata,
            request: result.request,
            response: result.response,
        };
    }

    async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
        const result = await this.src.doStream(v3CallOptionsToV4(options));
        return {
            stream: mapStream(result.stream, v4StreamPartToV3),
            request: result.request,
            response: result.response,
        };
    }
}

export const COMPAT = {
    v2: (v4: LanguageModelV4) => new V2Adapter(v4),
    v3: (v4: LanguageModelV4) => new V3Adapter(v4),
    v4: (v4: LanguageModelV4) => v4,
};
