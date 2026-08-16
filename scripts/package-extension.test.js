import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { afterEach, describe, expect, it } from 'vitest';
import { createDeterministicZip } from './package-extension.mjs';

const temporaryRoots = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

describe('deterministic extension package', () => {
  it('uses stable order, metadata, paths, and bytes without an external zip command', async () => {
    const root = await mkdtemp(join(tmpdir(), 'realimage-package-'));
    temporaryRoots.push(root);
    await mkdir(join(root, 'nested'));
    await writeFile(join(root, 'z.txt'), 'last');
    await writeFile(join(root, 'nested', 'a.txt'), 'first');

    const first = await createDeterministicZip(root);
    const second = await createDeterministicZip(root);
    expect(first).toEqual(second);
    expect(first.readUInt32LE(0)).toBe(0x04034b50);
    expect(first.subarray(-22).readUInt32LE(0)).toBe(0x06054b50);
    expect(readStoredEntries(first)).toEqual([
      { name: 'nested/a.txt', text: 'first', dosTime: 0, dosDate: 0x0021 },
      { name: 'z.txt', text: 'last', dosTime: 0, dosDate: 0x0021 }
    ]);
  });
});

function readStoredEntries(archive) {
  const entries = [];
  let offset = 0;
  while (archive.readUInt32LE(offset) === 0x04034b50) {
    const method = archive.readUInt16LE(offset + 8);
    const dosTime = archive.readUInt16LE(offset + 10);
    const dosDate = archive.readUInt16LE(offset + 12);
    const length = archive.readUInt32LE(offset + 22);
    const nameLength = archive.readUInt16LE(offset + 26);
    const extraLength = archive.readUInt16LE(offset + 28);
    expect(method).toBe(0);
    const nameStart = offset + 30;
    const dataStart = nameStart + nameLength + extraLength;
    entries.push({
      name: archive.subarray(nameStart, nameStart + nameLength).toString('utf8'),
      text: archive.subarray(dataStart, dataStart + length).toString('utf8'),
      dosTime,
      dosDate
    });
    offset = dataStart + length;
  }
  return entries;
}
