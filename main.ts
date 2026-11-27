// png-squeeze: smallest lossless PNG re-encoder for 1280x720 screenshots.
//
// One program answers a whole set: it reads /input/<id>/image.png for every
// id, and writes /output/<id>/image.png, byte-identical pixels, in as few
// bytes as it can. How it squeezes:
//
//   1. Colour-type reduction. A screenshot is rarely true RGBA: flat shots
//      hold at most 256 distinct colours (palette, 1 byte a pixel), grey
//      shots hold one channel (grey, or grey+alpha), opaque rich shots drop
//      the alpha channel (RGB), and a shot whose only non-opaque pixels are
//      one fully transparent colour drops the channel entirely and keys that
//      colour with tRNS instead.
//   2. Per-row filter search. Every row gets the filter (None/Sub/Up/Average/
//      Paeth) that compresses best, chosen by three cheap heuristics
//      (minimum absolute sum, fewest distinct byte pairs) and by brute force
//      (each filter actually deflated against the 32 KiB written before it).
//      Each candidate form is then weighed by a full zlib pass at level 6,
//      PLTE/tRNS chunk cost included, and the smallest wins.
//   3. Final deflate. The winning filtered stream is re-compressed under
//      several zlib settings (level 9, memLevel 9, filtered strategy, fixed
//      Huffman) and the smallest stream is kept.
//
// Everything is deterministic: no clock, no randomness, no environment read
// beyond the input/output directory overrides the kit's starter documents.
// Node 22, built-ins only, TypeScript erasable syntax only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { deflateSync, inflateSync, constants } from 'node:zlib';

// ---------------------------------------------------------------------------
// PNG chunk plumbing: CRC-32, chunk assembly, IHDR.
// ---------------------------------------------------------------------------

const SIG = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);

const CRC_TAB: Int32Array = (() => {
  const tab = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    tab[n] = c;
  }
  return tab;
})();

function crc32(buf: Uint8Array, from: number, to: number): number {
  let c = -1;
  for (let i = from; i < to; i++) c = CRC_TAB[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

function assemble(chunks: Array<[string, Uint8Array]>): Uint8Array {
  let total = SIG.length;
  for (const [, data] of chunks) total += 12 + data.length;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  out.set(SIG, 0);
  let at = SIG.length;
  for (const [type, data] of chunks) {
    view.setUint32(at, data.length);
    for (let i = 0; i < 4; i++) out[at + 4 + i] = type.charCodeAt(i);
    out.set(data, at + 8);
    view.setUint32(at + 8 + data.length, crc32(out, at + 4, at + 8 + data.length));
    at += 12 + data.length;
  }
  return out;
}

function ihdr(width: number, height: number, colourType: number): Uint8Array {
  const b = new Uint8Array(13);
  const v = new DataView(b.buffer);
  v.setUint32(0, width);
  v.setUint32(4, height);
  b[8] = 8; // bit depth
  b[9] = colourType;
  b[10] = 0; // compression
  b[11] = 0; // filter method
  b[12] = 0; // no interlace
  return b;
}

// ---------------------------------------------------------------------------
// PNG decoding to RGBA (bit depth 8, colour types 0/2/3/4/6, filters 0-4).
// ---------------------------------------------------------------------------

function paethPredict(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

interface Decoded {
  rgba: Uint8Array;
  width: number;
  height: number;
}

function decodePng(bytes: Uint8Array): Decoded {
  if (bytes.length < 8) throw new Error('file too short for a PNG signature');
  for (let i = 0; i < 8; i++) if (bytes[i] !== SIG[i]) throw new Error('not a PNG');
  let at = 8;
  let width = 0;
  let height = 0;
  let colourType = 6;
  let plte: Uint8Array | null = null;
  let trns: Uint8Array | null = null;
  const idat: Uint8Array[] = [];
  let idatLen = 0;
  let sawIend = false;
  while (at + 8 <= bytes.length) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const len = view.getUint32(at);
    const type = String.fromCharCode(bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]);
    const data = bytes.subarray(at + 8, at + 8 + len);
    if (type === 'IHDR') {
      width = view.getUint32(at + 8);
      height = view.getUint32(at + 12);
      if (data[8] !== 8) throw new Error('only bit depth 8 is supported');
      colourType = data[9];
      if (data[12] !== 0) throw new Error('interlaced PNGs are not supported');
    } else if (type === 'PLTE') {
      plte = data;
    } else if (type === 'tRNS') {
      trns = data;
    } else if (type === 'IDAT') {
      idat.push(data);
      idatLen += len;
    } else if (type === 'IEND') {
      sawIend = true;
      break;
    } else {
      throw new Error('unsupported chunk ' + type);
    }
    at += 12 + len;
  }
  if (!sawIend || width === 0 || height === 0) throw new Error('truncated PNG');
  const channels = colourType === 0 ? 1 : colourType === 2 ? 3 : colourType === 3 ? 1 : colourType === 4 ? 2 : 4;
  const bpp = colourType === 3 ? 1 : channels;
  const zipped = new Uint8Array(idatLen);
  let off = 0;
  for (const part of idat) {
    zipped.set(part, off);
    off += part.length;
  }
  const raw = inflateSync(zipped);
  const stride = width * channels;
  if (raw.length !== height * (stride + 1)) throw new Error('inflated size mismatch');
  // Unfilter in place, row by row, honouring each row's filter byte.
  const samples = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const src = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const dst = samples.subarray(y * stride, (y + 1) * stride);
    const up = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    for (let x = 0; x < stride; x++) {
      const left = x >= bpp ? dst[x - bpp] : 0;
      const above = up ? up[x] : 0;
      const upLeft = up && x >= bpp ? up[x - bpp] : 0;
      let v = src[x];
      if (filter === 1) v = (v + left) & 0xff;
      else if (filter === 2) v = (v + above) & 0xff;
      else if (filter === 3) v = (v + ((left + above) >> 1)) & 0xff;
      else if (filter === 4) v = (v + paethPredict(left, above, upLeft)) & 0xff;
      else if (filter !== 0) throw new Error('unknown filter ' + filter);
      dst[x] = v;
    }
  }
  // Expand samples to RGBA.
  const rgba = new Uint8Array(width * height * 4);
  for (let i = 0, s = 0, d = 0; i < width * height; i++, d += 4) {
    if (colourType === 0) {
      const g = samples[s++];
      rgba[d] = g;
      rgba[d + 1] = g;
      rgba[d + 2] = g;
      rgba[d + 3] = trns && trns.length >= 2 && g === trns[1] ? 0 : 255;
    } else if (colourType === 2) {
      const r = samples[s++];
      const g = samples[s++];
      const b = samples[s++];
      rgba[d] = r;
      rgba[d + 1] = g;
      rgba[d + 2] = b;
      rgba[d + 3] =
        trns && trns.length >= 6 && r === trns[1] && g === trns[3] && b === trns[5] ? 0 : 255;
    } else if (colourType === 3) {
      const idx = samples[s++];
      if (!plte || idx * 3 + 2 >= plte.length) throw new Error('palette index out of range');
      rgba[d] = plte[idx * 3];
      rgba[d + 1] = plte[idx * 3 + 1];
      rgba[d + 2] = plte[idx * 3 + 2];
      rgba[d + 3] = trns && idx < trns.length ? trns[idx] : 255;
    } else if (colourType === 4) {
      const g = samples[s++];
      rgba[d] = g;
      rgba[d + 1] = g;
      rgba[d + 2] = g;
      rgba[d + 3] = samples[s++];
    } else {
      rgba[d] = samples[s++];
      rgba[d + 1] = samples[s++];
      rgba[d + 2] = samples[s++];
      rgba[d + 3] = samples[s++];
    }
  }
  return { rgba, width, height };
}

// ---------------------------------------------------------------------------
// Analysis: opacity, greyness, and the colour census (up to 256 colours).
// ---------------------------------------------------------------------------

interface Colour {
  r: number;
  g: number;
  b: number;
  a: number;
  word: number;
  count: number;
}

interface Analysis {
  opaque: boolean;
  grey: boolean;
  colours: Colour[] | null;
}

function analyse(rgba: Uint8Array): Analysis {
  const n = rgba.length >>> 2;
  let opaque = true;
  let grey = true;
  const census = new Map<number, number>();
  let overflow = false;
  for (let i = 0, d = 0; i < n; i++, d += 4) {
    if (rgba[d + 3] !== 255) opaque = false;
    if (rgba[d] !== rgba[d + 1] || rgba[d] !== rgba[d + 2]) grey = false;
    if (!overflow) {
      const word = (rgba[d] | (rgba[d + 1] << 8) | (rgba[d + 2] << 16) | (rgba[d + 3] << 24)) >>> 0;
      const seen = census.get(word);
      if (seen === undefined) {
        if (census.size === 256) overflow = true;
        else census.set(word, 1);
      } else {
        census.set(word, seen + 1);
      }
    }
  }
  if (overflow) return { opaque, grey, colours: null };
  const colours: Colour[] = [];
  for (const [word, count] of census) {
    colours.push({
      r: word & 0xff,
      g: (word >>> 8) & 0xff,
      b: (word >>> 16) & 0xff,
      a: (word >>> 24) & 0xff,
      word,
      count,
    });
  }
  return { opaque, grey, colours };
}

// One fully transparent colour that no opaque pixel shares, so the image can
// drop its alpha channel and key that colour with tRNS instead. Null when the
// image has partial alpha or several transparent colours.
function transparentKey(rgba: Uint8Array, grey: boolean): number | null {
  let key = -1;
  for (let d = 0; d < rgba.length; d += 4) {
    const a = rgba[d + 3];
    if (a === 255) continue;
    if (a !== 0) return null;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (key === -1) key = v;
    else if (key !== v) return null;
  }
  if (key === -1) return null;
  for (let d = 0; d < rgba.length; d += 4) {
    if (rgba[d + 3] !== 255) continue;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (v === key) return null;
  }
  return key;
}

// ---------------------------------------------------------------------------
// Candidate forms: palette, grey, grey+alpha, RGB, RGBA.
// ---------------------------------------------------------------------------

interface Form {
  colourType: number;
  bpp: number;
  samples: Uint8Array;
  plte: Uint8Array | null;
  trns: Uint8Array | null;
}

function luminance(c: Colour): number {
  return 299 * c.r + 587 * c.g + 114 * c.b;
}

// Palette entries: transparent ones first (by alpha), then by `order`.
function paletteForm(rgba: Uint8Array, colours: Colour[], order: 'count' | 'luminance'): Form {
  const entries = [...colours].sort((x, y) => {
    const tx = x.a === 255 ? 1 : 0;
    const ty = y.a === 255 ? 1 : 0;
    if (tx !== ty) return tx - ty;
    if (tx === 0 && x.a !== y.a) return x.a - y.a;
    if (order === 'luminance') return luminance(x) - luminance(y) || x.word - y.word;
    return y.count - x.count || x.word - y.word;
  });
  const index = new Map<number, number>();
  entries.forEach((c, i) => index.set(c.word, i));
  const plte = new Uint8Array(entries.length * 3);
  let lastTransparent = 0;
  entries.forEach((c, i) => {
    plte[i * 3] = c.r;
    plte[i * 3 + 1] = c.g;
    plte[i * 3 + 2] = c.b;
    if (c.a !== 255) lastTransparent = i + 1;
  });
  const trns = lastTransparent === 0 ? null : Uint8Array.from(entries.slice(0, lastTransparent), (c) => c.a);
  const n = rgba.length >>> 2;
  const indices = new Uint8Array(n);
  for (let i = 0, d = 0; i < n; i++, d += 4) {
    const word = (rgba[d] | (rgba[d + 1] << 8) | (rgba[d + 2] << 16) | (rgba[d + 3] << 24)) >>> 0;
    indices[i] = index.get(word) as number;
  }
  return { colourType: 3, bpp: 1, samples: indices, plte, trns };
}

function channelForm(rgba: Uint8Array, colourType: number): Form {
  const n = rgba.length >>> 2;
  const ch = colourType === 0 ? 1 : colourType === 4 ? 2 : colourType === 2 ? 3 : 4;
  if (colourType === 6) return { colourType, bpp: 4, samples: rgba, plte: null, trns: null };
  const out = new Uint8Array(n * ch);
  for (let i = 0, d = 0, s = 0; i < n; i++, d += 4, s += ch) {
    if (colourType === 0) out[s] = rgba[d];
    else if (colourType === 4) {
      out[s] = rgba[d];
      out[s + 1] = rgba[d + 3];
    } else {
      out[s] = rgba[d];
      out[s + 1] = rgba[d + 1];
      out[s + 2] = rgba[d + 2];
    }
  }
  return { colourType, bpp: ch, samples: out, plte: null, trns: null };
}

function formsOf(rgba: Uint8Array, info: Analysis): Form[] {
  const forms: Form[] = [];
  if (info.colours !== null) {
    forms.push(paletteForm(rgba, info.colours, 'count'));
    forms.push(paletteForm(rgba, info.colours, 'luminance'));
    if (!info.grey) return forms;
  }
  const key = info.opaque ? null : transparentKey(rgba, info.grey);
  let colourType: number;
  let trns: Uint8Array | null = null;
  if (info.grey) {
    colourType = info.opaque || key !== null ? 0 : 4;
    if (key !== null) trns = Uint8Array.of(0, key);
  } else {
    colourType = info.opaque || key !== null ? 2 : 6;
    if (key !== null) {
      trns = Uint8Array.of(0, (key >>> 16) & 0xff, 0, (key >>> 8) & 0xff, 0, key & 0xff);
    }
  }
  const form = channelForm(rgba, colourType);
  form.trns = trns;
  forms.push(form);
  return forms;
}

// ---------------------------------------------------------------------------
// PNG filters, per row.
// ---------------------------------------------------------------------------

function filterRow(type: number, cur: Uint8Array, prev: Uint8Array | null, bpp: number, out: Uint8Array): void {
  for (let x = 0; x < cur.length; x++) {
    const left = x >= bpp ? cur[x - bpp] : 0;
    const above = prev ? prev[x] : 0;
    const upLeft = prev && x >= bpp ? prev[x - bpp] : 0;
    let v = cur[x];
    if (type === 1) v -= left;
    else if (type === 2) v -= above;
    else if (type === 3) v -= (left + above) >> 1;
    else if (type === 4) v -= paethPredict(left, above, upLeft);
    out[x] = v & 0xff;
  }
}

// Sum of absolute values, reading each byte as signed (-128..127).
function absSum(row: Uint8Array): number {
  let s = 0;
  for (let i = 0; i < row.length; i++) {
    const v = row[i];
    s += v < 128 ? v : 256 - v;
  }
  return s;
}

// How many distinct neighbouring byte pairs a row holds (fewer = more repeats).
function pairCount(row: Uint8Array, seen: Uint32Array, stamp: number): number {
  let n = 0;
  for (let i = 1; i < row.length; i++) {
    const pair = (row[i - 1] << 8) | row[i];
    if (seen[pair] !== stamp) {
      seen[pair] = stamp;
      n++;
    }
  }
  return n;
}

type FilterChoice = number | 'minsum' | 'pairs' | 'brute';

const CONTEXT = 32768; // bytes of filtered data a brute-force row is tried after

function filteredRows(
  samples: Uint8Array,
  width: number,
  height: number,
  bpp: number,
  choice: FilterChoice,
): Uint8Array {
  const stride = width * bpp;
  const out = new Uint8Array(height * (stride + 1));
  const trials: Uint8Array[] = [];
  for (let t = 0; t < 5; t++) {
    const row = new Uint8Array(stride + 1);
    row[0] = t;
    trials.push(row);
  }
  const seen = new Uint32Array(65536);
  let stamp = 0;
  for (let y = 0; y < height; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    const at = y * (stride + 1);
    if (typeof choice === 'number') {
      out[at] = choice;
      filterRow(choice, cur, prev, bpp, out.subarray(at + 1, at + 1 + stride));
      continue;
    }
    const before = out.subarray(Math.max(0, at - CONTEXT), at);
    let type = 0;
    let best = Infinity;
    for (let t = 0; t < 5; t++) {
      filterRow(t, cur, prev, bpp, trials[t].subarray(1));
      let cost: number;
      if (choice === 'minsum') cost = absSum(trials[t].subarray(1));
      else if (choice === 'pairs') cost = pairCount(trials[t].subarray(1), seen, ++stamp);
      else {
        cost = deflateSync(trials[t], {
          level: 4,
          memLevel: 9,
          dictionary: before.length === 0 ? undefined : before,
        }).length;
      }
      if (cost < best) {
        best = cost;
        type = t;
      }
    }
    out.set(trials[type], at);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Screening: weigh every form x filter choice with a full zlib pass, keep the
// smallest (chunk overhead included). Then re-compress the winner under
// several zlib settings and keep the smallest stream.
// ---------------------------------------------------------------------------

const SCREEN_OPTS = { level: 6, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY };

function chunkOverhead(form: Form): number {
  return (form.plte === null ? 0 : 12 + form.plte.length) + (form.trns === null ? 0 : 12 + form.trns.length);
}

function pngFrom(form: Form, width: number, height: number, idat: Uint8Array): Uint8Array {
  const chunks: Array<[string, Uint8Array]> = [['IHDR', ihdr(width, height, form.colourType)]];
  if (form.plte !== null) chunks.push(['PLTE', form.plte]);
  if (form.trns !== null) chunks.push(['tRNS', form.trns]);
  chunks.push(['IDAT', idat]);
  chunks.push(['IEND', new Uint8Array(0)]);
  return assemble(chunks);
}

// ---------------------------------------------------------------------------
// Optimal-parse DEFLATE (a small zopfli-style encoder, written from the
// DEFLATE format itself, not from any existing compressor's code).
//
// Node's zlib picks matches greedily; an optimal parser instead scores every
// position with dynamic programming: at each byte it weighs emitting a
// literal against every match length the hash chains found, using the true
// Huffman cost of each choice, and keeps the cheapest path to the end. Two
// passes run: first under fixed-Huffman costs, then under dynamic-Huffman
// costs rebuilt from the first parse's symbol frequencies (and once more
// from the second's), and the smallest emitted stream wins.
// ---------------------------------------------------------------------------

const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

// Length symbol (257..285) and extra bits for a match length.
function lenSymbol(len: number): [number, number, number] {
  if (len <= 10) return [257 + (len - 3), 0, 0];
  for (let j = 28; j >= 8; j--) {
    if (len >= LEN_BASE[j]) return [257 + j, LEN_EXTRA[j], len - LEN_BASE[j]];
  }
  return [257, 0, 0];
}

// Distance symbol (0..29) and extra bits for a match distance.
function distSymbol(dist: number): [number, number, number] {
  for (let j = 29; j >= 0; j--) {
    if (dist >= DIST_BASE[j]) return [j, DIST_EXTRA[j], dist - DIST_BASE[j]];
  }
  return [0, 0, 0];
}

// Fixed-Huffman code lengths (RFC 1951 §3.2.6).
function fixedLitLen(sym: number): number {
  if (sym < 144) return 8;
  if (sym < 256) return 9;
  if (sym < 280) return 7;
  return 8;
}

// Canonical Huffman codes from code lengths; returns bit-reversed codes
// ready to emit LSB-first, plus the lengths themselves.
function canonicalCodes(lengths: ArrayLike<number>): { codes: Int32Array; lens: Uint8Array } {
  const n = lengths.length;
  const lens = new Uint8Array(n);
  for (let i = 0; i < n; i++) lens[i] = lengths[i];
  const maxLen = Math.max(0, ...Array.from(lens));
  const count = new Int32Array(maxLen + 1);
  for (let i = 0; i < n; i++) if (lens[i] > 0) count[lens[i]]++;
  const next = new Int32Array(maxLen + 1);
  let code = 0;
  for (let b = 1; b <= maxLen; b++) {
    code = (code + count[b - 1]) << 1;
    next[b] = code;
  }
  const codes = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const l = lens[i];
    if (l === 0) continue;
    // Reverse the canonical code's bits: DEFLATE emits Huffman codes
    // starting with their most significant bit.
    let c = next[l]++;
    let rev = 0;
    for (let b = 0; b < l; b++) {
      rev = (rev << 1) | (c & 1);
      c >>>= 1;
    }
    codes[i] = rev;
  }
  return { codes, lens };
}

class BitWriter {
  bytes: number[] = [];
  bitBuf = 0;
  bitLen = 0;
  putBits(value: number, len: number): void {
    this.bitBuf |= (value & ((1 << len) - 1)) << this.bitLen;
    this.bitLen += len;
    while (this.bitLen >= 8) {
      this.bytes.push(this.bitBuf & 0xff);
      this.bitBuf >>>= 8;
      this.bitLen -= 8;
    }
  }
  putCode(code: number, len: number): void {
    this.putBits(code, len);
  }
  // Total bits written so far (unpadded): compare candidates without
  // finishing them, since only the final block may be byte-padded.
  bitLength(): number {
    return this.bytes.length * 8 + this.bitLen;
  }
  // Append another writer's unpadded bits after this writer's bits.
  append(other: BitWriter): void {
    for (const b of other.bytes) this.putBits(b, 8);
    if (other.bitLen > 0) this.putBits(other.bitBuf, other.bitLen);
  }
  finish(): Uint8Array {
    if (this.bitLen > 0) this.bytes.push(this.bitBuf & 0xff);
    return Uint8Array.from(this.bytes);
  }
}

function adler32Of(data: Uint8Array): number {
  let a = 1;
  let b = 0;
  for (let i = 0; i < data.length; ) {
    const end = Math.min(data.length, i + 5552);
    for (; i < end; i++) {
      a += data[i];
      b += a;
    }
    a %= 65521;
    b %= 65521;
  }
  return ((b << 16) | a) >>> 0;
}

// Length-limited Huffman lengths from symbol frequencies (max 15 bits),
// zlib-style: build an ordinary Huffman tree, then if any depth exceeds the
// limit, rebalance the length counts (move one leaf down, take two up) until
// the overflow is gone, and reassign the shortest lengths to the most
// frequent symbols. Returns lengths, or null when no symbol is used.
function limitedLengths(freq: ArrayLike<number>, maxLen: number): Uint8Array | null {
  const n = freq.length;
  let total = 0;
  for (let i = 0; i < n; i++) total += freq[i];
  if (total === 0) return null;
  // Heap of node indices; nodes beyond n are internal, with combined freq.
  const nodeFreq: number[] = Array.from(freq);
  const parent = new Int32Array(2 * n).fill(-1);
  const depth = new Int32Array(2 * n);
  const heap: number[] = [];
  for (let i = 0; i < n; i++) if (freq[i] > 0) heap.push(i);
  const less = (a: number, b: number): boolean =>
    nodeFreq[a] < nodeFreq[b] || (nodeFreq[a] === nodeFreq[b] && a < b);
  const heapPush = (v: number): void => {
    heap.push(v);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0];
    const last = heap.pop() as number;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && less(heap[l], heap[m])) m = l;
        if (r < heap.length && less(heap[r], heap[m])) m = r;
        if (m === i) break;
        [heap[i], heap[m]] = [heap[m], heap[i]];
        i = m;
      }
    }
    return top;
  };
  // Heapify the initial leaves.
  for (let i = (heap.length >> 1) - 1; i >= 0; i--) {
    let k = i;
    for (;;) {
      const l = k * 2 + 1;
      const r = l + 1;
      let m = k;
      if (l < heap.length && less(heap[l], heap[m])) m = l;
      if (r < heap.length && less(heap[r], heap[m])) m = r;
      if (m === k) break;
      [heap[k], heap[m]] = [heap[m], heap[k]];
      k = m;
    }
  }
  let nextNode = n;
  while (heap.length > 1) {
    const a = heapPop();
    const b = heapPop();
    nodeFreq[nextNode] = nodeFreq[a] + nodeFreq[b];
    parent[a] = nextNode;
    parent[b] = nextNode;
    heapPush(nextNode);
    nextNode++;
  }
  const root = heap.length === 1 ? heap[0] : -1;
  if (root === -1) return null;
  // Depths by walking parents.
  const rawLen = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    if (freq[i] === 0) continue;
    let d = 0;
    let p = i;
    while (parent[p] !== -1) {
      d++;
      p = parent[p];
    }
    rawLen[i] = d;
    if (d === 0) d = 1; // a single symbol still needs one bit
    if (d > maxLen) d = maxLen; // clamp; the Kraft repair below pays for it
    rawLen[i] = d;
  }
  // Repair the Kraft sum: clamping overlong codes can oversubscribe the
  // code (sum 2^-len > 1). Lengthen the rarest symbols that still have
  // room, one bit at a time, until the sum is back at or under 1. This
  // always terminates: with every length at the limit the sum is at most
  // n * 2^-maxLen <= 1 for every code this encoder builds (n <= 2^maxLen).
  const lengths = new Uint8Array(n);
  for (let i = 0; i < n; i++) lengths[i] = rawLen[i];
  const kraft = (): number => {
    let s = 0;
    for (let i = 0; i < n; i++) if (lengths[i] > 0) s += Math.pow(2, -lengths[i]);
    return s;
  };
  while (kraft() > 1) {
    // The rarest symbol with room to grow.
    let pick = -1;
    let pickFreq = Infinity;
    for (let i = 0; i < n; i++) {
      if (freq[i] > 0 && lengths[i] > 0 && lengths[i] < maxLen && freq[i] < pickFreq) {
        pickFreq = freq[i];
        pick = i;
      }
    }
    if (pick === -1) break; // every length is at the limit: sum <= 1 already
    lengths[pick]++;
  }
  return lengths;
}

// Candidate match lengths for a longest match of `m`: the top of each
// length symbol's range, plus `m` itself — the only lengths where the
// encoded cost can change. Built once, shared by every parse.
const CANDS: number[][] = (() => {
  const table: number[][] = [];
  for (let m = 0; m <= 258; m++) {
    const list: number[] = [];
    for (let j = 0; j < 29; j++) {
      const hi = Math.min(m, LEN_BASE[j] + (1 << LEN_EXTRA[j]) - 1);
      if (hi >= LEN_BASE[j] && hi >= 3 && (list.length === 0 || list[list.length - 1] !== hi)) list.push(hi);
    }
    if (m >= 3 && (list.length === 0 || list[list.length - 1] !== m)) list.push(m);
    table.push(list);
  }
  return table;
})();

// The longest match at every position, found once and reused by all three
// DP passes (the matcher does not depend on the Huffman costs). Hash chains
// of 3-byte sequences, built forward: each position is inserted into its
// chain only after it is searched, so the chain always holds positions
// strictly before the search point and the walk is a short one (at most 32
// steps, stopping only on a maximal match). Every position is searched:
// skipping positions inside a long match would hand the DP a match that
// shrinks by one each step, while the run continuing there is still 258
// long, and the DP pays for the difference with extra matches.
function findMatches(data: Uint8Array): { lens: Uint16Array; dists: Int32Array } {
  const N = data.length;
  const HASH_SIZE = 1 << 16;
  const head = new Int32Array(HASH_SIZE).fill(-1);
  const prev = new Int32Array(N).fill(-1);
  const lens = new Uint16Array(N);
  const dists = new Int32Array(N);
  const MAX_CHAIN = 24;
  const GOOD = 192; // a match this long is already cheap per byte; deeper
  // candidates rarely buy enough to pay for their extension cost
  for (let i = 0; i + 2 < N; i++) {
    const h = (((data[i] << 10) ^ (data[i + 1] << 5) ^ data[i + 2]) & (HASH_SIZE - 1)) >>> 0;
    let cand = head[h];
    const limit = Math.min(258, N - i);
    let longest = 0;
    let longestDist = 0;
    let steps = 0;
    while (cand !== -1 && i - cand <= 32768 && steps < MAX_CHAIN) {
      steps++;
      let m = 0;
      while (m < limit && data[cand + m] === data[i + m]) m++;
      if (m > longest) {
        longest = m;
        longestDist = i - cand;
        if (m >= GOOD || m === limit) break;
      }
      cand = prev[cand];
    }
    if (longest >= 3) {
      lens[i] = longest;
      dists[i] = longestDist;
    }
    prev[i] = head[h];
    head[h] = i;
  }
  return { lens, dists };
}

// One optimal parse of `data` under the given per-symbol bit costs, reusing
// the matches found by findMatches. Returns the sequence of (length,
// distance) choices: length 0 = literal.
function optimalParse(
  data: Uint8Array,
  matches: { lens: Uint16Array; dists: Int32Array },
  litCost: ArrayLike<number>,
  lenCost: ArrayLike<number>,
  distCost: ArrayLike<number>,
): { lens: Uint16Array; dists: Int32Array } {
  const N = data.length;
  const cost = new Float64Array(N + 1);
  const choiceLen = new Uint16Array(N);
  const choiceDist = new Int32Array(N);
  cost[N] = 0;
  for (let i = N - 1; i >= 0; i--) {
    let best = litCost[data[i]] + cost[i + 1];
    let bestLen = 0;
    let bestDist = 0;
    const longest = matches.lens[i];
    if (longest >= 3) {
      const dist = matches.dists[i];
      const dc = distCost[dist];
      const cands = CANDS[longest];
      for (let k = 0; k < cands.length; k++) {
        const L = cands[k];
        const c = lenCost[L] + dc + cost[i + L];
        if (c < best) {
          best = c;
          bestLen = L;
          bestDist = dist;
        }
      }
    }
    cost[i] = best;
    choiceLen[i] = bestLen;
    choiceDist[i] = bestDist;
  }
  return { lens: choiceLen, dists: choiceDist };
}

// Bit-cost tables for one Huffman model: literal costs (288), match-length
// costs indexed by length (0..258), distance costs indexed by distance.
function costTables(
  litLens: ArrayLike<number>,
  distLens: ArrayLike<number>,
): { lit: Float64Array; len: Float64Array; dist: Float64Array } {
  const lit = new Float64Array(288);
  for (let s = 0; s < 288; s++) lit[s] = litLens[s];
  const len = new Float64Array(259);
  for (let L = 3; L <= 258; L++) {
    const [ls, le] = lenSymbol(L);
    len[L] = litLens[ls] + le;
  }
  const dist = new Float64Array(32769);
  for (let d = 1; d <= 32768; d++) {
    const [ds, de] = distSymbol(d);
    dist[d] = distLens[ds] + de;
  }
  return { lit, len, dist };
}

// Symbol frequencies of a parse (lit/len symbols 0..287, dist symbols 0..29).
function parseFrequencies(
  data: Uint8Array,
  parse: { lens: Uint16Array; dists: Int32Array },
): { litFreq: Int32Array; distFreq: Int32Array } {
  const litFreq = new Int32Array(288);
  const distFreq = new Int32Array(30);
  let i = 0;
  while (i < data.length) {
    const L = parse.lens[i];
    if (L === 0) {
      litFreq[data[i]]++;
      i++;
    } else {
      const [ls] = lenSymbol(L);
      litFreq[ls]++;
      const [ds] = distSymbol(parse.dists[i]);
      distFreq[ds]++;
      i += L;
    }
  }
  litFreq[256]++; // end of block
  return { litFreq, distFreq };
}

// Emit one dynamic-Huffman block for a parse.
function emitDynamic(
  w: BitWriter,
  data: Uint8Array,
  parse: { lens: Uint16Array; dists: Int32Array },
  litLens: Uint8Array,
  distLens: Uint8Array,
  isFinal: boolean,
): void {
  const { codes: litCodes } = canonicalCodes(litLens);
  const { codes: distCodes } = canonicalCodes(distLens);
  // Trim trailing zero lengths (keep at least 257 lit codes, 1 dist code).
  let hlit = 288;
  while (hlit > 257 && litLens[hlit - 1] === 0) hlit--;
  let hdist = 30;
  while (hdist > 1 && distLens[hdist - 1] === 0) hdist--;
  // Run-length encode the lit+dist length sequence (symbols 0..18).
  const seq: number[] = [];
  for (let i = 0; i < hlit; i++) seq.push(litLens[i]);
  for (let i = 0; i < hdist; i++) seq.push(distLens[i]);
  const clSyms: number[] = [];
  const clExtra: Array<[number, number]> = []; // [value, bits] per emitted symbol
  let i = 0;
  while (i < seq.length) {
    const v = seq[i];
    let run = 1;
    while (i + run < seq.length && seq[i + run] === v) run++;
    if (v === 0) {
      let left = run;
      while (left >= 11) {
        const take = Math.min(left, 138);
        clSyms.push(18);
        clExtra.push([take - 11, 7]);
        left -= take;
      }
      while (left >= 3) {
        const take = Math.min(left, 10);
        clSyms.push(17);
        clExtra.push([take - 3, 3]);
        left -= take;
      }
      while (left > 0) {
        clSyms.push(0);
        clExtra.push([0, 0]);
        left--;
      }
    } else {
      clSyms.push(v);
      clExtra.push([0, 0]);
      let left = run - 1;
      while (left >= 3) {
        const take = Math.min(left, 6);
        clSyms.push(16);
        clExtra.push([take - 3, 2]);
        left -= take;
      }
      while (left > 0) {
        clSyms.push(v);
        clExtra.push([0, 0]);
        left--;
      }
    }
    i += run;
  }
  const clFreq = new Int32Array(19);
  for (const s of clSyms) clFreq[s]++;
  let clLens = limitedLengths(clFreq, 7);
  if (clLens === null) clLens = new Uint8Array(19);
  // A one-symbol code-length code is not a valid prefix code; give a second
  // symbol length 1 as well.
  let used = 0;
  let onlySym = -1;
  for (let s = 0; s < 19; s++) {
    if (clLens[s] > 0) {
      used++;
      onlySym = s;
    }
  }
  if (used === 1) {
    const dummy = onlySym === 0 ? 1 : 0;
    clLens[dummy] = 1;
  }
  const { codes: clCodes } = canonicalCodes(clLens);
  const ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
  let hclen = 19;
  while (hclen > 4 && clLens[ORDER[hclen - 1]] === 0) hclen--;
  w.putBits(isFinal ? 1 : 0, 1); // BFINAL
  w.putBits(2, 2); // BTYPE = 10 dynamic
  w.putBits(hlit - 257, 5);
  w.putBits(hdist - 1, 5);
  w.putBits(hclen - 4, 4);
  for (let k = 0; k < hclen; k++) w.putBits(clLens[ORDER[k]], 3);
  for (let k = 0; k < clSyms.length; k++) {
    const s = clSyms[k];
    w.putCode(clCodes[s], clLens[s]);
    const [val, bits] = clExtra[k];
    if (bits > 0) w.putBits(val, bits);
  }
  // The data itself.
  let p = 0;
  while (p < data.length) {
    const L = parse.lens[p];
    if (L === 0) {
      const sym = data[p];
      w.putCode(litCodes[sym], litLens[sym]);
      p++;
    } else {
      const [ls, le, lv] = lenSymbol(L);
      w.putCode(litCodes[ls], litLens[ls]);
      if (le > 0) w.putBits(lv, le);
      const [ds, de, dv] = distSymbol(parse.dists[p]);
      w.putCode(distCodes[ds], distLens[ds]);
      if (de > 0) w.putBits(dv, de);
      p += L;
    }
  }
  w.putCode(litCodes[256], litLens[256]);
}

// Emit one fixed-Huffman block for a parse.
function emitFixed(
  w: BitWriter,
  data: Uint8Array,
  parse: { lens: Uint16Array; dists: Int32Array },
  isFinal: boolean,
): void {
  const litLens = new Uint8Array(288);
  for (let s = 0; s < 288; s++) litLens[s] = fixedLitLen(s);
  const distLens = new Uint8Array(30).fill(5);
  const { codes: litCodes } = canonicalCodes(litLens);
  const { codes: distCodes } = canonicalCodes(distLens);
  w.putBits(isFinal ? 1 : 0, 1); // BFINAL
  w.putBits(1, 2); // BTYPE = 01 fixed
  let p = 0;
  while (p < data.length) {
    const L = parse.lens[p];
    if (L === 0) {
      const sym = data[p];
      w.putCode(litCodes[sym], litLens[sym]);
      p++;
    } else {
      const [ls, le, lv] = lenSymbol(L);
      w.putCode(litCodes[ls], litLens[ls]);
      if (le > 0) w.putBits(lv, le);
      const [ds, de, dv] = distSymbol(parse.dists[p]);
      w.putCode(distCodes[ds], distLens[ds]);
      if (de > 0) w.putBits(dv, de);
      p += L;
    }
  }
  w.putCode(litCodes[256], litLens[256]);
}

// Wrap raw DEFLATE bytes in a zlib stream.
function zlibWrap(raw: Uint8Array, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(2 + raw.length + 4);
  out[0] = 0x78;
  out[1] = 0xda;
  out.set(raw, 2);
  const adler = adler32Of(data);
  out[2 + raw.length] = (adler >>> 24) & 0xff;
  out[2 + raw.length + 1] = (adler >>> 16) & 0xff;
  out[2 + raw.length + 2] = (adler >>> 8) & 0xff;
  out[2 + raw.length + 3] = adler & 0xff;
  return out;
}

// Encode one DEFLATE block (fixed pass, then two dynamic passes) and
// return the smallest writer for it. Only the stream's last block is
// final; the caller pads and wraps.
function encodeBlock(data: Uint8Array, isFinal: boolean): BitWriter | null {
  try {
    const matches = findMatches(data);
    const fixedLit = new Float64Array(288);
    for (let s = 0; s < 288; s++) fixedLit[s] = fixedLitLen(s);
    const fixedDist = new Float64Array(30).fill(5);
    const fixed = costTables(fixedLit, fixedDist);
    const parse0 = optimalParse(data, matches, fixed.lit, fixed.len, fixed.dist);
    const w0 = new BitWriter();
    emitFixed(w0, data, parse0, isFinal);
    let best = w0;
    // Dynamic passes: each round parses under the previous round's costs,
    // then rebuilds the code lengths from the NEW parse's frequencies and
    // emits with those — so every symbol the stream emits genuinely has a
    // code. A symbol the previous parse never used costs 15 bits in the
    // DP, never 0: a free literal would corrupt the stream.
    let freq = parseFrequencies(data, parse0);
    let litLens = limitedLengths(freq.litFreq, 15);
    let distLens = limitedLengths(freq.distFreq, 15);
    for (let round = 0; round < 2; round++) {
      if (litLens === null || distLens === null) break;
      if (distLens.every((v) => v === 0)) distLens[0] = 1;
      const costLit = new Float64Array(288);
      for (let s = 0; s < 288; s++) costLit[s] = litLens[s] === 0 ? 15 : litLens[s];
      const costDist = new Float64Array(30);
      for (let s = 0; s < 30; s++) costDist[s] = distLens[s] === 0 ? 15 : distLens[s];
      const tables = costTables(costLit, costDist);
      const parse = optimalParse(data, matches, tables.lit, tables.len, tables.dist);
      const freq2 = parseFrequencies(data, parse);
      const lit2 = limitedLengths(freq2.litFreq, 15);
      const dist2 = limitedLengths(freq2.distFreq, 15);
      if (lit2 === null || dist2 === null) break;
      if (dist2.every((v) => v === 0)) dist2[0] = 1;
      const w = new BitWriter();
      emitDynamic(w, data, parse, lit2, dist2, isFinal);
      if (w.bitLength() < best.bitLength()) best = w;
      litLens = lit2;
      distLens = dist2;
    }
    return best;
  } catch {
    return null;
  }
}

// The optimal-parse stream for `data`: split into row-aligned blocks — a
// screenshot's regions compress differently, so each block gets its own
// Huffman code, worth a few percent at no extra time since the work is
// proportional to the data — encode each block at its best, and wrap the
// blocks in one zlib stream. Returns null on any internal failure.
function optimalDeflate(data: Uint8Array, stride = 0): Uint8Array | null {
  try {
    const rows = stride > 0 ? Math.floor(data.length / stride) : 0;
    const blockCount = rows >= 16 ? 8 : 1;
    const out = new BitWriter();
    for (let b = 0; b < blockCount; b++) {
      const start = Math.floor((rows * b) / blockCount) * stride;
      const end = b === blockCount - 1 ? data.length : Math.floor((rows * (b + 1)) / blockCount) * stride;
      const block = blockCount === 1 ? data : data.subarray(start, end);
      const w = encodeBlock(block, b === blockCount - 1);
      if (w === null) return null;
      out.append(w);
    }
    return zlibWrap(out.finish(), data);
  } catch {
    return null;
  }
}

function squeeze(rgba: Uint8Array, width: number, height: number): Uint8Array {
  const info = analyse(rgba);
  const forms = formsOf(rgba, info);
  let bestForm: Form | null = null;
  let bestData: Uint8Array | null = null;
  let bestBytes = Infinity;
  for (const form of forms) {
    const isPalette = form.colourType === 3;
    const choices: FilterChoice[] = isPalette
      ? form.plte !== null && form.trns === null
        ? [0]
        : [0, 'minsum', 'brute']
      : [0, 1, 2, 4, 'minsum', 'pairs', 'brute'];
    // Palette-by-count is screened unfiltered only; the luminance palette
    // gets the full search. Tell them apart by order of creation: the first
    // palette form is the count one.
    const paletteIndex = isPalette ? forms.indexOf(form) : -1;
    const useChoices = isPalette && paletteIndex === 0 ? [0 as FilterChoice] : choices;
    const overhead = chunkOverhead(form);
    for (const choice of useChoices) {
      const data = filteredRows(form.samples, width, height, form.bpp, choice);
      const z = deflateSync(data, SCREEN_OPTS);
      const bytes = z.length + overhead;
      if (bytes < bestBytes) {
        bestBytes = bytes;
        bestForm = form;
        bestData = data;
      }
    }
  }
  if (bestForm === null || bestData === null) throw new Error('no form screened');
  // Final deflate: several zlib settings on the winning filtered stream,
  // plus the optimal-parse encoder; the smallest stream is kept.
  const finals = [
    { level: 9, memLevel: 9, strategy: constants.Z_DEFAULT_STRATEGY },
    { level: 9, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY },
    { level: 9, memLevel: 9, strategy: constants.Z_FILTERED },
    { level: 9, memLevel: 9, strategy: constants.Z_FIXED },
    { level: 9, memLevel: 9, strategy: constants.Z_RLE },
  ];
  let bestStream: Uint8Array | null = null;
  for (const opts of finals) {
    const z = deflateSync(bestData, opts);
    if (bestStream === null || z.length < bestStream.length) bestStream = z;
  }
  // The optimal parse loses to zlib on palette data (runs of 0/1 indices
  // are zlib's home turf), so only run it on non-palette forms.
  // The optimal parse loses to zlib on palette data (runs of 0/1 indices
  // are zlib's home turf), so only run it on non-palette forms.
  const optimal =
    bestForm.colourType === 3 ? null : optimalDeflate(bestData, bestData.length / height);
  if (optimal !== null && optimal.length < (bestStream as Uint8Array).length) bestStream = optimal;
  return pngFrom(bestForm, width, height, bestStream as Uint8Array);
}

// ---------------------------------------------------------------------------
// Entry point: one run answers every instance in the input directory.
// ---------------------------------------------------------------------------

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

for (const id of readdirSync(INPUT).sort()) {
  const image = decodePng(new Uint8Array(readFileSync(join(INPUT, id, 'image.png'))));
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'image.png'), squeeze(image.rgba, image.width, image.height));
}
