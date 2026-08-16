const dataset = process.argv[2] || 'Zitacron/real-vs-ai-corpus';
const end = Number(process.argv[3] || 180_000);
const step = Number(process.argv[4] || 500);
const concurrency = Number(process.env.PROOFMARK_FETCH_CONCURRENCY || 16);
const offsets = Array.from({ length: Math.ceil(end / step) }, (_, index) => index * step);
const samples = [];

for (let cursor = 0; cursor < offsets.length; cursor += concurrency) {
  const batch = offsets.slice(cursor, cursor + concurrency);
  samples.push(...await Promise.all(batch.map(async (offset) => {
    const parameters = new URLSearchParams({ dataset, config: 'default', split: 'train', offset: String(offset), length: '1' });
    const response = await fetchWithRetry(`https://datasets-server.huggingface.co/rows?${parameters}`);
    const payload = await response.json();
    const item = payload.rows?.[0];
    if (!item) throw new Error(`No row returned at offset ${offset}`);
    return { offset, label: Number(item.row.label), source: item.row.source_dataset };
  })));
  process.stderr.write(`\rMapped ${Math.min(cursor + batch.length, offsets.length)}/${offsets.length}`);
}

process.stderr.write('\n');
samples.sort((a, b) => a.offset - b.offset);
let previous;
for (const sample of samples) {
  if (!previous || sample.source !== previous.source || sample.label !== previous.label) {
    console.log(`${String(sample.offset).padStart(9)}  ${sample.label === 1 ? 'ai  ' : 'real'}  ${sample.source}`);
  }
  previous = sample;
}

async function fetchWithRetry(url) {
  let lastError;
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
      lastError = new Error(`HTTP ${response.status} for ${url}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, attempt * 400));
  }
  throw lastError;
}
