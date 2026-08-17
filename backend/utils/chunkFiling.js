import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const ITEM_HEADER_RE = /^#{0,6}\s*item\s+(\d+[a-z]?)\b/i;
// NOTE: no underscore in the sentinel — Turndown escapes literal "_" in text
// content (it's markdown italic syntax) to "\_", which silently breaks any
// exact-string match against the marker. Confirmed via repro: with
// underscores, 0/64 real tables survived Turndown; without them, all did.
const TABLE_PROSE_RE = /§TABLEPROSESTART§([\s\S]*?)§TABLEPROSEEND§/g;

// SEC filings mention every "Item N" twice: once in the table of contents,
// once as the real section start. We keep only the LAST occurrence of each
// item number as the real boundary and fold everything before it into a
// "Preamble" bucket (TOC, cover page, etc). Validated against a real NVDA
// 10-K — produced a clean, ordered, deduped Item 1 -> Item 16 list.
function splitIntoSections(markdown) {
  const lines = markdown.split("\n");
  const matches = [];

  lines.forEach((line, i) => {
    const m = line.trim().match(ITEM_HEADER_RE);
    if (m) matches.push({ lineIndex: i, itemNum: m[1].toUpperCase() });
  });

  if (matches.length === 0) {
    return [{ section: "Full Filing", text: markdown }];
  }

  const lastOccurrence = new Map();
  for (const m of matches) lastOccurrence.set(m.itemNum, m.lineIndex);
  const boundaries = matches.filter((m) => lastOccurrence.get(m.itemNum) === m.lineIndex);

  const sections = [];
  if (boundaries[0].lineIndex > 0) {
    sections.push({
      section: "Preamble",
      text: lines.slice(0, boundaries[0].lineIndex).join("\n"),
    });
  }
  for (let i = 0; i < boundaries.length; i++) {
    const start = boundaries[i].lineIndex;
    const end = i + 1 < boundaries.length ? boundaries[i + 1].lineIndex : lines.length;
    sections.push({
      section: `Item ${boundaries[i].itemNum}`,
      text: lines.slice(start, end).join("\n"),
    });
  }
  return sections;
}

// Table-prose blocks are atomic at the ROW level, not the whole-table level.
// tableToProse already emits one complete sentence per row, joined by "\n".
// A single financial-statement table can produce 30K+ chars of prose, which
// blows past Cohere embed-english-v3.0's ~512 token input limit if kept as
// one chunk. So: group whole rows together up to maxChars, never splitting
// a row's sentence mid-way. Confirmed via NVDA FY2025 run: the 36,351-char
// Item 15 table was the trigger for this.
const TABLE_CHUNK_MAX_CHARS = 1500;

function splitTableProse(text, maxChars = TABLE_CHUNK_MAX_CHARS) {
  const rows = text
    .split("\n")
    .map((r) => r.trim())
    .filter(Boolean);

  if (rows.length === 0) return [];

  const chunks = [];
  let current = [];
  let currentLen = 0;

  for (const row of rows) {
    const rowLen = row.length + 1; // +1 accounts for the joining "\n"
    if (current.length > 0 && currentLen + rowLen > maxChars) {
      chunks.push(current.join("\n"));
      current = [];
      currentLen = 0;
    }
    if (row.length > maxChars) {
      // a single row alone exceeds maxChars — extremely rare (one giant
      // sentence), but row atomicity wins over size here. Flush it as its
      // own oversized chunk rather than truncating data.
      if (current.length > 0) {
        chunks.push(current.join("\n"));
        current = [];
        currentLen = 0;
      }
      chunks.push(row);
      continue;
    }
    current.push(row);
    currentLen += rowLen;
  }
  if (current.length > 0) chunks.push(current.join("\n"));

  return chunks;
}

// Splits a section's text into an ordered sequence of narrative / table-prose
// blocks. Table-prose blocks are returned whole — never size-split,
// regardless of length (atomic-block rule).
function extractAtomicBlocks(text) {
  const blocks = [];
  let lastIndex = 0;
  let match;

  TABLE_PROSE_RE.lastIndex = 0;
  while ((match = TABLE_PROSE_RE.exec(text)) !== null) {
    const before = text.slice(lastIndex, match.index);
    if (before.trim()) blocks.push({ type: "narrative", text: before });

    const tableText = match[1].replace(/ {2}\n|\\\n/g, "\n").trim();
    if (tableText) blocks.push({ type: "table-prose", text: tableText });

    lastIndex = TABLE_PROSE_RE.lastIndex;
  }
  const rest = text.slice(lastIndex);
  if (rest.trim()) blocks.push({ type: "narrative", text: rest });

  return blocks;
}

// baseMetadata is expected to already contain { company, year, formType,
// filingDate } — "year" (not "fiscalYear") to match the fixed schema the
// fine-tuned router/ragNode filter on. xbrlFacts entries carry their own
// fiscalYear from xbrlFacts.js internally; we remap that to `year` here too
// so every chunk in the index — narrative, table-prose, xbrl-fact — shares
// one consistent field name for Pinecone metadata filtering.
export async function chunkFiling(markdown, xbrlFacts, baseMetadata) {
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const chunks = [];

  const sections = splitIntoSections(markdown);

  for (const { section, text } of sections) {
    const blocks = extractAtomicBlocks(text);

    for (const block of blocks) {
      if (block.type === "table-prose") {
        const tableChunks = splitTableProse(block.text);
        for (const tableChunkText of tableChunks) {
          chunks.push({
            pageContent: tableChunkText,
            metadata: { ...baseMetadata, section, sourceType: "table-prose" },
          });
        }
      } else {
        const narrativeChunks = await splitter.splitText(block.text);
        for (const chunkText of narrativeChunks) {
          if (!chunkText.trim()) continue;
          chunks.push({
            pageContent: chunkText,
            metadata: { ...baseMetadata, section, sourceType: "narrative" },
          });
        }
      }
    }
  }

  for (const fact of xbrlFacts) {
    chunks.push({
      pageContent: fact.text,
      metadata: {
        ...baseMetadata,
        section: "xbrl-facts",
        sourceType: "xbrl-fact",
        year: fact.fiscalYear, // overrides baseMetadata.year with the fact's own FY
      },
    });
  }

  return chunks;
}