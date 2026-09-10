import { rm } from 'node:fs/promises';

await rm('./dist', { recursive: true, force: true });

const entrypoints = ['./index.ts', './v2.ts', './v3.ts', './v4.ts'];

const external = [
    '@ai-sdk/provider',
    '@ai-sdk/provider-utils',
    'workers-ai-provider',
    'zod',
    'ai',
];

await Bun.build({
    entrypoints,
    external,
    sourcemap: 'linked',
    minify: true,
    target: 'node',
    outdir: './dist',
    format: 'esm',
    naming: '[dir]/[name].js',
});

await Bun.$`./node_modules/.bin/tsc -p tsconfig.build.json`;

console.log('built dist/');
