import { z } from 'zod';
import type {
    JSONArray,
    JSONObject,
    JSONSchema7,
    JSONValue,
    LanguageModelV2CallOptions,
    LanguageModelV3CallOptions,
    LanguageModelV4CallOptions,
    SharedV2ProviderOptions,
    SharedV3ProviderOptions,
    SharedV4ProviderReference,
} from '@ai-sdk/provider';

// ---------------------------------------------------------------------------
// Shared building blocks (identical shape across versions)
// ---------------------------------------------------------------------------

/** Any JSON value. Comes from `JSON.parse`, so no further validation needed. */
const jsonValueSchema: z.ZodType<JSONValue> = z.custom<JSONValue>();

/** A JSON schema as defined by the `json-schema` package (recursive, opaque). */
const jsonSchemaSchema: z.ZodType<JSONSchema7> = z.custom<JSONSchema7>(
    (v) => typeof v === 'object' && v !== null && !Array.isArray(v),
);

/** A dotted provider identifier like `provider.tool-name`. */
const dottedStringSchema: z.ZodType<`${string}.${string}`> = z.custom<`${string}.${string}`>(
    (v) => typeof v === 'string' && v.includes('.'),
);

/** `Record<string, string | undefined>` — additional HTTP headers. */
const headersSchema = z.record(z.string(), z.union([z.string(), z.undefined()]));

// z.custom keeps the exact `Uint8Array` type; z.instanceof narrows to `Uint8Array<ArrayBuffer>`.
const uint8ArraySchema: z.ZodType<Uint8Array> = z.custom<Uint8Array>((v) => v instanceof Uint8Array);

/** `Uint8Array | string | URL` — V2/V3 file data. */
const dataContentSchema = z.union([uint8ArraySchema, z.string(), z.instanceof(URL)]);

/** Response format, identical in v2/v3/v4. */
const responseFormatSchema = z.union([
    z.object({ type: z.literal('text') }),
    z.object({
        type: z.literal('json'),
        schema: jsonSchemaSchema.optional(),
        name: z.string().optional(),
        description: z.string().optional(),
    }),
]);

/** Tool choice, identical in v2/v3/v4. */
const toolChoiceSchema = z.union([
    z.object({ type: z.literal('auto') }),
    z.object({ type: z.literal('none') }),
    z.object({ type: z.literal('required') }),
    z.object({ type: z.literal('tool'), toolName: z.string() }),
]);

// ---------------------------------------------------------------------------
// V2
// ---------------------------------------------------------------------------

const v2ProviderOptionsSchema: z.ZodType<SharedV2ProviderOptions> = z.record(
    z.string(),
    z.record(z.string(), jsonValueSchema),
);

const v2TextPartSchema = z.object({
    type: z.literal('text'),
    text: z.string(),
    providerOptions: v2ProviderOptionsSchema.optional(),
});

const v2FilePartSchema = z.object({
    type: z.literal('file'),
    filename: z.string().optional(),
    data: dataContentSchema,
    mediaType: z.string(),
    providerOptions: v2ProviderOptionsSchema.optional(),
});

const v2ReasoningPartSchema = z.object({
    type: z.literal('reasoning'),
    text: z.string(),
    providerOptions: v2ProviderOptionsSchema.optional(),
});

const v2ToolCallPartSchema = z.object({
    type: z.literal('tool-call'),
    toolCallId: z.string(),
    toolName: z.string(),
    input: z.unknown(),
    providerExecuted: z.boolean().optional(),
    providerOptions: v2ProviderOptionsSchema.optional(),
});

const v2ToolResultOutputSchema = z.union([
    z.object({ type: z.literal('text'), value: z.string() }),
    z.object({ type: z.literal('json'), value: jsonValueSchema }),
    z.object({ type: z.literal('error-text'), value: z.string() }),
    z.object({ type: z.literal('error-json'), value: jsonValueSchema }),
    z.object({
        type: z.literal('content'),
        value: z.array(
            z.union([
                z.object({ type: z.literal('text'), text: z.string() }),
                z.object({ type: z.literal('media'), data: z.string(), mediaType: z.string() }),
            ]),
        ),
    }),
]);

const v2ToolResultPartSchema = z.object({
    type: z.literal('tool-result'),
    toolCallId: z.string(),
    toolName: z.string(),
    output: v2ToolResultOutputSchema,
    providerOptions: v2ProviderOptionsSchema.optional(),
});

const v2MessageSchema = z.union([
    z.object({
        role: z.literal('system'),
        content: z.string(),
        providerOptions: v2ProviderOptionsSchema.optional(),
    }),
    z.object({
        role: z.literal('user'),
        content: z.array(z.union([v2TextPartSchema, v2FilePartSchema])),
        providerOptions: v2ProviderOptionsSchema.optional(),
    }),
    z.object({
        role: z.literal('assistant'),
        content: z.array(
            z.union([
                v2TextPartSchema,
                v2FilePartSchema,
                v2ReasoningPartSchema,
                v2ToolCallPartSchema,
                v2ToolResultPartSchema,
            ]),
        ),
        providerOptions: v2ProviderOptionsSchema.optional(),
    }),
    z.object({
        role: z.literal('tool'),
        content: z.array(v2ToolResultPartSchema),
        providerOptions: v2ProviderOptionsSchema.optional(),
    }),
]);

const v2FunctionToolSchema = z.object({
    type: z.literal('function'),
    name: z.string(),
    description: z.string().optional(),
    inputSchema: jsonSchemaSchema,
    providerOptions: v2ProviderOptionsSchema.optional(),
});

const v2ProviderDefinedToolSchema = z.object({
    type: z.literal('provider-defined'),
    id: dottedStringSchema,
    name: z.string(),
    args: z.record(z.string(), z.unknown()),
});

export const v2callOptionsSchema = z.object({
    prompt: z.array(v2MessageSchema),
    maxOutputTokens: z.number().optional(),
    temperature: z.number().optional(),
    stopSequences: z.array(z.string()).optional(),
    topP: z.number().optional(),
    topK: z.number().optional(),
    presencePenalty: z.number().optional(),
    frequencyPenalty: z.number().optional(),
    responseFormat: responseFormatSchema.optional(),
    seed: z.number().optional(),
    tools: z.array(z.union([v2FunctionToolSchema, v2ProviderDefinedToolSchema])).optional(),
    toolChoice: toolChoiceSchema.optional(),
    includeRawChunks: z.boolean().optional(),
    abortSignal: z.instanceof(AbortSignal).optional(),
    headers: headersSchema.optional(),
    providerOptions: v2ProviderOptionsSchema.optional(),
});

// ---------------------------------------------------------------------------
// V3
// ---------------------------------------------------------------------------

const v3ProviderOptionsSchema: z.ZodType<SharedV3ProviderOptions> = z.record(
    z.string(),
    z.record(z.string(), z.union([jsonValueSchema, z.undefined()])),
);

const v3TextPartSchema = z.object({
    type: z.literal('text'),
    text: z.string(),
    providerOptions: v3ProviderOptionsSchema.optional(),
});

const v3FilePartSchema = z.object({
    type: z.literal('file'),
    filename: z.string().optional(),
    data: dataContentSchema,
    mediaType: z.string(),
    providerOptions: v3ProviderOptionsSchema.optional(),
});

const v3ReasoningPartSchema = z.object({
    type: z.literal('reasoning'),
    text: z.string(),
    providerOptions: v3ProviderOptionsSchema.optional(),
});

const v3ToolCallPartSchema = z.object({
    type: z.literal('tool-call'),
    toolCallId: z.string(),
    toolName: z.string(),
    input: z.unknown(),
    providerExecuted: z.boolean().optional(),
    providerOptions: v3ProviderOptionsSchema.optional(),
});

const v3FileIdSchema = z.union([z.string(), z.record(z.string(), z.string())]);

const v3ToolResultContentSchema = z.array(
    z.union([
        z.object({
            type: z.literal('text'),
            text: z.string(),
            providerOptions: v3ProviderOptionsSchema.optional(),
        }),
        z.object({
            type: z.literal('file-data'),
            data: z.string(),
            mediaType: z.string(),
            filename: z.string().optional(),
            providerOptions: v3ProviderOptionsSchema.optional(),
        }),
        z.object({
            type: z.literal('file-url'),
            url: z.string(),
            providerOptions: v3ProviderOptionsSchema.optional(),
        }),
        z.object({
            type: z.literal('file-id'),
            fileId: v3FileIdSchema,
            providerOptions: v3ProviderOptionsSchema.optional(),
        }),
        z.object({
            type: z.literal('image-data'),
            data: z.string(),
            mediaType: z.string(),
            providerOptions: v3ProviderOptionsSchema.optional(),
        }),
        z.object({
            type: z.literal('image-url'),
            url: z.string(),
            providerOptions: v3ProviderOptionsSchema.optional(),
        }),
        z.object({
            type: z.literal('image-file-id'),
            fileId: v3FileIdSchema,
            providerOptions: v3ProviderOptionsSchema.optional(),
        }),
        z.object({
            type: z.literal('custom'),
            providerOptions: v3ProviderOptionsSchema.optional(),
        }),
    ]),
);

const v3ToolResultOutputSchema = z.union([
    z.object({
        type: z.literal('text'),
        value: z.string(),
        providerOptions: v3ProviderOptionsSchema.optional(),
    }),
    z.object({
        type: z.literal('json'),
        value: jsonValueSchema,
        providerOptions: v3ProviderOptionsSchema.optional(),
    }),
    z.object({
        type: z.literal('execution-denied'),
        reason: z.string().optional(),
        providerOptions: v3ProviderOptionsSchema.optional(),
    }),
    z.object({
        type: z.literal('error-text'),
        value: z.string(),
        providerOptions: v3ProviderOptionsSchema.optional(),
    }),
    z.object({
        type: z.literal('error-json'),
        value: jsonValueSchema,
        providerOptions: v3ProviderOptionsSchema.optional(),
    }),
    z.object({
        type: z.literal('content'),
        value: v3ToolResultContentSchema,
    }),
]);

const v3ToolResultPartSchema = z.object({
    type: z.literal('tool-result'),
    toolCallId: z.string(),
    toolName: z.string(),
    output: v3ToolResultOutputSchema,
    providerOptions: v3ProviderOptionsSchema.optional(),
});

const v3ToolApprovalResponsePartSchema = z.object({
    type: z.literal('tool-approval-response'),
    approvalId: z.string(),
    approved: z.boolean(),
    reason: z.string().optional(),
    providerOptions: v3ProviderOptionsSchema.optional(),
});

const v3MessageSchema = z.union([
    z.object({
        role: z.literal('system'),
        content: z.string(),
        providerOptions: v3ProviderOptionsSchema.optional(),
    }),
    z.object({
        role: z.literal('user'),
        content: z.array(z.union([v3TextPartSchema, v3FilePartSchema])),
        providerOptions: v3ProviderOptionsSchema.optional(),
    }),
    z.object({
        role: z.literal('assistant'),
        content: z.array(
            z.union([
                v3TextPartSchema,
                v3FilePartSchema,
                v3ReasoningPartSchema,
                v3ToolCallPartSchema,
                v3ToolResultPartSchema,
            ]),
        ),
        providerOptions: v3ProviderOptionsSchema.optional(),
    }),
    z.object({
        role: z.literal('tool'),
        content: z.array(z.union([v3ToolResultPartSchema, v3ToolApprovalResponsePartSchema])),
        providerOptions: v3ProviderOptionsSchema.optional(),
    }),
]);

const v3FunctionToolSchema = z.object({
    type: z.literal('function'),
    name: z.string(),
    description: z.string().optional(),
    inputSchema: jsonSchemaSchema,
    inputExamples: z.array(z.object({ input: z.record(z.string(), z.union([jsonValueSchema, z.undefined()])) })).optional(),
    strict: z.boolean().optional(),
    providerOptions: v3ProviderOptionsSchema.optional(),
});

const v3ProviderToolSchema = z.object({
    type: z.literal('provider'),
    id: dottedStringSchema,
    name: z.string(),
    args: z.record(z.string(), z.unknown()),
});

export const v3callOptionsSchema = z.object({
    prompt: z.array(v3MessageSchema),
    maxOutputTokens: z.number().optional(),
    temperature: z.number().optional(),
    stopSequences: z.array(z.string()).optional(),
    topP: z.number().optional(),
    topK: z.number().optional(),
    presencePenalty: z.number().optional(),
    frequencyPenalty: z.number().optional(),
    responseFormat: responseFormatSchema.optional(),
    seed: z.number().optional(),
    tools: z.array(z.union([v3FunctionToolSchema, v3ProviderToolSchema])).optional(),
    toolChoice: toolChoiceSchema.optional(),
    includeRawChunks: z.boolean().optional(),
    abortSignal: z.instanceof(AbortSignal).optional(),
    headers: headersSchema.optional(),
    providerOptions: v3ProviderOptionsSchema.optional(),
});

// ---------------------------------------------------------------------------
// V4
// ---------------------------------------------------------------------------

const v4ProviderOptionsSchema = v3ProviderOptionsSchema;

const v4FileDataDataSchema = z.object({
    type: z.literal('data'),
    data: z.union([uint8ArraySchema, z.string()]),
});

const v4FileDataUrlSchema = z.object({
    type: z.literal('url'),
    url: z.instanceof(URL),
});

const v4FileDataReferenceSchema = z.object({
    type: z.literal('reference'),
    reference: z.custom<SharedV4ProviderReference>(),
});

const v4FileDataTextSchema = z.object({
    type: z.literal('text'),
    text: z.string(),
});

const v4FileDataSchema = z.union([
    v4FileDataDataSchema,
    v4FileDataUrlSchema,
    v4FileDataReferenceSchema,
    v4FileDataTextSchema,
]);

const v4TextPartSchema = z.object({
    type: z.literal('text'),
    text: z.string(),
    providerOptions: v4ProviderOptionsSchema.optional(),
});

const v4FilePartSchema = z.object({
    type: z.literal('file'),
    filename: z.string().optional(),
    data: v4FileDataSchema,
    mediaType: z.string(),
    providerOptions: v4ProviderOptionsSchema.optional(),
});

const v4CustomPartSchema = z.object({
    type: z.literal('custom'),
    kind: dottedStringSchema,
    providerOptions: v4ProviderOptionsSchema.optional(),
});

const v4ReasoningPartSchema = z.object({
    type: z.literal('reasoning'),
    text: z.string(),
    providerOptions: v4ProviderOptionsSchema.optional(),
});

const v4ReasoningFilePartSchema = z.object({
    type: z.literal('reasoning-file'),
    data: z.union([v4FileDataDataSchema, v4FileDataUrlSchema]),
    mediaType: z.string(),
    providerOptions: v4ProviderOptionsSchema.optional(),
});

const v4ToolCallPartSchema = z.object({
    type: z.literal('tool-call'),
    toolCallId: z.string(),
    toolName: z.string(),
    input: z.unknown(),
    providerExecuted: z.boolean().optional(),
    providerOptions: v4ProviderOptionsSchema.optional(),
});

const v4ToolResultContentSchema = z.array(
    z.union([
        z.object({
            type: z.literal('text'),
            text: z.string(),
            providerOptions: v4ProviderOptionsSchema.optional(),
        }),
        z.object({
            type: z.literal('file'),
            data: v4FileDataSchema,
            mediaType: z.string(),
            filename: z.string().optional(),
            providerOptions: v4ProviderOptionsSchema.optional(),
        }),
        z.object({
            type: z.literal('custom'),
            providerOptions: v4ProviderOptionsSchema.optional(),
        }),
    ]),
);

const v4ToolResultOutputSchema = z.union([
    z.object({
        type: z.literal('text'),
        value: z.string(),
        providerOptions: v4ProviderOptionsSchema.optional(),
    }),
    z.object({
        type: z.literal('json'),
        value: jsonValueSchema,
        providerOptions: v4ProviderOptionsSchema.optional(),
    }),
    z.object({
        type: z.literal('execution-denied'),
        reason: z.string().optional(),
        providerOptions: v4ProviderOptionsSchema.optional(),
    }),
    z.object({
        type: z.literal('error-text'),
        value: z.string(),
        providerOptions: v4ProviderOptionsSchema.optional(),
    }),
    z.object({
        type: z.literal('error-json'),
        value: jsonValueSchema,
        providerOptions: v4ProviderOptionsSchema.optional(),
    }),
    z.object({
        type: z.literal('content'),
        value: v4ToolResultContentSchema,
    }),
]);

const v4ToolResultPartSchema = z.object({
    type: z.literal('tool-result'),
    toolCallId: z.string(),
    toolName: z.string(),
    output: v4ToolResultOutputSchema,
    providerOptions: v4ProviderOptionsSchema.optional(),
});

const v4ToolApprovalResponsePartSchema = z.object({
    type: z.literal('tool-approval-response'),
    approvalId: z.string(),
    approved: z.boolean(),
    reason: z.string().optional(),
    providerOptions: v4ProviderOptionsSchema.optional(),
});

const v4MessageSchema = z.union([
    z.object({
        role: z.literal('system'),
        content: z.string(),
        providerOptions: v4ProviderOptionsSchema.optional(),
    }),
    z.object({
        role: z.literal('user'),
        content: z.array(z.union([v4TextPartSchema, v4FilePartSchema])),
        providerOptions: v4ProviderOptionsSchema.optional(),
    }),
    z.object({
        role: z.literal('assistant'),
        content: z.array(
            z.union([
                v4TextPartSchema,
                v4FilePartSchema,
                v4CustomPartSchema,
                v4ReasoningPartSchema,
                v4ReasoningFilePartSchema,
                v4ToolCallPartSchema,
                v4ToolResultPartSchema,
            ]),
        ),
        providerOptions: v4ProviderOptionsSchema.optional(),
    }),
    z.object({
        role: z.literal('tool'),
        content: z.array(z.union([v4ToolResultPartSchema, v4ToolApprovalResponsePartSchema])),
        providerOptions: v4ProviderOptionsSchema.optional(),
    }),
]);

const v4FunctionToolSchema = z.object({
    type: z.literal('function'),
    name: z.string(),
    description: z.string().optional(),
    inputSchema: jsonSchemaSchema,
    inputExamples: z.array(z.object({ input: z.record(z.string(), z.union([jsonValueSchema, z.undefined()])) })).optional(),
    strict: z.boolean().optional(),
    providerOptions: v4ProviderOptionsSchema.optional(),
});

const v4ProviderToolSchema = z.object({
    type: z.literal('provider'),
    id: dottedStringSchema,
    name: z.string(),
    args: z.record(z.string(), z.unknown()),
});

export const v4callOptionsSchema = z.object({
    prompt: z.array(v4MessageSchema),
    maxOutputTokens: z.number().optional(),
    temperature: z.number().optional(),
    stopSequences: z.array(z.string()).optional(),
    topP: z.number().optional(),
    topK: z.number().optional(),
    presencePenalty: z.number().optional(),
    frequencyPenalty: z.number().optional(),
    responseFormat: responseFormatSchema.optional(),
    seed: z.number().optional(),
    tools: z.array(z.union([v4FunctionToolSchema, v4ProviderToolSchema])).optional(),
    toolChoice: toolChoiceSchema.optional(),
    includeRawChunks: z.boolean().optional(),
    abortSignal: z.instanceof(AbortSignal).optional(),
    headers: headersSchema.optional(),
    reasoning: z.enum(['provider-default', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh']).optional(),
    providerOptions: v4ProviderOptionsSchema.optional(),
});

type Schema<TVersion extends 'v2' | 'v3' | 'v4'> = TVersion extends 'v2'
    ? typeof v2callOptionsSchema
    : TVersion extends 'v3'
      ? typeof v3callOptionsSchema
      : TVersion extends 'v4'
        ? typeof v4callOptionsSchema
        : never;

export function getSchema<TVersion extends 'v2' | 'v3' | 'v4'>(version: TVersion): Schema<TVersion> {
    switch (version) {
        case 'v2':
            return v2callOptionsSchema as Schema<TVersion>;
        case 'v3':
            return v3callOptionsSchema as Schema<TVersion>;
        case 'v4':
            return v4callOptionsSchema as Schema<TVersion>;
    }
}

type _Check<TSchema, TOptions> =
    z.infer<TSchema> extends TOptions ? (TOptions extends z.infer<TSchema> ? true : never) : never;
// Compile-time tripwire: if @ai-sdk/provider's type changes shape,
// this line stops compiling instead of silently drifting.

const _assertInSyncv2: _Check<typeof v2callOptionsSchema, LanguageModelV2CallOptions> = true;
const _assertInSyncv3: _Check<typeof v3callOptionsSchema, LanguageModelV3CallOptions> = true;
const _assertInSyncv4: _Check<typeof v4callOptionsSchema, LanguageModelV4CallOptions> = true;
