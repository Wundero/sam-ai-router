import { z } from 'zod';

export const providerSchema = z.enum(['cloudflare', 'deepseek', 'moonshot', 'zhipu']);

export const decisionSchema = z.object({
    model: z.string(),
    provider: providerSchema,
});

export type Provider = z.output<typeof providerSchema>;
export type Decision = z.output<typeof decisionSchema>;
