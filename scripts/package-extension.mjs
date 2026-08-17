import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const DEFAULT_INPUT = resolve('dist');
const DEFAULT_ARTIFACT = resolve('artifacts/realimage-v1.1.0.zip');
const CRC32_TABLE = makeCrc32Table();

export async function createDeterministicZip(inputRoot = DEFAULT_INPUT) {
  const files = await collectFiles(resolve(inputRoot));
  if (!files.length) throw new Error(`No files found below ${resolve(inputRoot)}.`);

  const localParts = [];
  const centralParts = [];
  let localOffset = 0;
  for (const file of files) {
    const name = Buffer.from(file.name, 'utf8');
    const data = await readFile(file.path);
    const checksum = crc32(data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0x0800, 6);
    local.writeUInt16LE(0, 8); // Stored: the large model is already compressed/quantized.
    local.writeUInt16LE(0, 10); // 00:00:00
    local.writeUInt16LE(0x0021, 12); // 1980-01-01
    local.writeUInt32LE(checksum, 14);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(name.length, 26);
    local.writeUInt16LE(0, 28);
    localParts.push(local, name, data);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE((3 << 8) | 20, 4); // Unix creator, ZIP 2.0.
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0x0800, 8);
    central.writeUInt16LE(0, 10);
    central.writeUInt16LE(0, 12);
    central.writeUInt16LE(0x0021, 14);
    central.writeUInt32LE(checksum, 16);
    central.writeUInt32LE(data.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt16LE(0, 30);
    central.writeUInt16LE(0, 32);
    central.writeUInt16LE(0, 34);
    central.writeUInt16LE(0, 36);
    central.writeUInt32LE((0o100644 << 16) >>> 0, 38);
    central.writeUInt32LE(localOffset, 42);
    centralParts.push(central, name);

    localOffset += local.length + name.length + data.length;
  }

  if (files.length > 0xffff) throw new Error('ZIP64 is not supported by this deterministic packager.');
  const centralDirectory = Buffer.concat(centralParts);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(0, 4);
  end.writeUInt16LE(0, 6);
  end.writeUInt16LE(files.length, 8);
  end.writeUInt16LE(files.length, 10);
  end.writeUInt32LE(centralDirectory.length, 12);
  end.writeUInt32LE(localOffset, 16);
  end.writeUInt16LE(0, 20);
  return Buffer.concat([...localParts, centralDirectory, end]);
}

export async function writeDeterministicPackage({
  inputRoot = DEFAULT_INPUT,
  artifact = DEFAULT_ARTIFACT
} = {}) {
  const archive = await createDeterministicZip(inputRoot);
  const target = resolve(artifact);
  const temporary = `${target}.tmp-${process.pid}`;
  await mkdir(dirname(target), { recursive: true });
  await writeFile(temporary, archive);
  await rm(target, { force: true });
  await rename(temporary, target);
  return {
    artifact: target,
    bytes: archive.length,
    sha256: createHash('sha256').update(archive).digest('hex')
  };
}

async function collectFiles(root, directory = root) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const path = resolve(directory, entry.name);
    if (entry.isSymbolicLink()) throw new Error(`Refusing to package symbolic link: ${path}`);
    if (entry.isDirectory()) files.push(...await collectFiles(root, path));
    else if (entry.isFile()) {
      const name = relative(root, path).replaceAll('\\', '/');
      if (!name || name.startsWith('../') || name.includes('/../')) throw new Error(`Unsafe ZIP path: ${name}`);
      files.push({ name, path });
    }
  }
  return files.sort((left, right) => left.name.localeCompare(right.name));
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = (crc >>> 8) ^ CRC32_TABLE[(crc ^ byte) & 0xff];
  return (crc ^ 0xffffffff) >>> 0;
}

function makeCrc32Table() {
  const table = new Uint32Array(256);
  for (let value = 0; value < table.length; value += 1) {
    let crc = value;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
    table[value] = crc >>> 0;
  }
  return table;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const result = await writeDeterministicPackage();
  console.log(`Created ${result.artifact}`);
  console.log(`Bytes: ${result.bytes}`);
  console.log(`SHA-256: ${result.sha256}`);
}
