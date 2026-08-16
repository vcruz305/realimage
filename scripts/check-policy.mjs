import { readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';

const tracked = execFileSync('git', ['ls-files', '--cached', '--others', '--exclude-standard', 'src', 'scripts'], { encoding: 'utf8' })
  .trim()
  .split('\n')
  .filter(Boolean)
  .filter((file) => /\.(?:js|mjs|html|css|json)$/.test(file));

const forbidden = [
  [/fetch\(\s*['\"]https?:/g, 'remote fetch'],
  [/XMLHttpRequest/g, 'XMLHttpRequest'],
  [/(?:openai|anthropic|replicate|fal\.ai|stability\.ai)\.com/gi, 'inference service reference']
];

const violations = [];
for (const file of tracked) {
  if (['scripts/download-model.mjs', 'scripts/check-policy.mjs'].includes(file)) continue;
  const source = await readFile(file, 'utf8');
  for (const [pattern, label] of forbidden) {
    if (pattern.test(source)) violations.push(`${file}: ${label}`);
    pattern.lastIndex = 0;
  }
}

if (violations.length) {
  console.error(`Privacy policy check failed:\n${violations.map((item) => `  - ${item}`).join('\n')}`);
  process.exit(1);
}
console.log(`Privacy policy check passed across ${tracked.length} source files.`);
