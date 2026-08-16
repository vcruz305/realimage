import { defineConfig } from 'vite';
import { crx } from '@crxjs/vite-plugin';
import { copyFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import manifest from './src/manifest.js';
import { LOCAL_RUNTIME, MODEL } from './src/shared/constants.js';

const RELEASE_DOCUMENTS = ['LICENSE', 'NOTICE.md', 'PRIVACY.md'];
const RELEASE_ASSETS = [
  ...['config.json', 'preprocessor_config.json', MODEL.weightFile].map((relativePath) => ({
    source: `public/models/${MODEL.id}/${relativePath}`,
    destination: `models/${MODEL.id}/${relativePath}`
  })),
  ...[LOCAL_RUNTIME.wasmLoader, LOCAL_RUNTIME.wasmModule].map((name) => ({
    source: `public/wasm/${name}`,
    destination: `wasm/${name}`
  }))
];

function copyReleaseFiles() {
  return {
    name: 'realimage-release-files',
    async writeBundle(outputOptions) {
      const outputDirectory = resolve(import.meta.dirname, outputOptions.dir || 'dist');
      const copies = [
        ...RELEASE_DOCUMENTS.map((name) => ({ source: name, destination: name })),
        ...RELEASE_ASSETS
      ];
      await Promise.all(copies.map(async ({ source, destination }) => {
        const target = resolve(outputDirectory, destination);
        await mkdir(dirname(target), { recursive: true });
        await copyFile(resolve(import.meta.dirname, source), target);
      }));
    }
  };
}

export default defineConfig({
  publicDir: false,
  plugins: [crx({ manifest }), copyReleaseFiles()],
  test: {
    exclude: ['**/_upstream/**', '**/node_modules/**', '**/dist/**']
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    target: 'es2022',
    rollupOptions: {
      input: {
        offscreen: resolve(import.meta.dirname, 'src/offscreen/offscreen.html')
      },
      output: {
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: (assetInfo) => assetInfo.name === 'ort-wasm-simd-threaded.jsep.wasm'
          ? 'wasm/[name][extname]'
          : 'assets/[name]-[hash][extname]'
      }
    }
  }
});
