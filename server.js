// server.js — SOL WhatsApp Assistant (Meta Webhook, multilingual, Vision OCR, KB + Embeddings)
// Version: 2025-11-09.r2
// -------------------------------------------------------------------------------------------
// Replit secrets required:
//   VERIFY_TOKEN, WHATSAPP_TOKEN, PHONE_NUMBER_ID, OPENAI_API_KEY
//   INDEX_URL, ICS_URL_BASE
// Optional:
//   OCR_API_KEY            (fallback OCR.Space)
//   OPENAI_MODEL           (default: gpt-4o-mini)
//   EMBEDDING_MODEL        (default: text-embedding-3-small)
// -------------------------------------------------------------------------------------------

import express from "express";
import axios from "axios";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";
import FormData from "form-data";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const VERSION = "2025-11-09.r2";

const app = express();
app.use(express.json({ limit: "25mb" }));

// === ENV ===
const {
  VERIFY_TOKEN,
  WHATSAPP_TOKEN,
  PHONE_NUMBER_ID,
  OPENAI_API_KEY,
  INDEX_URL = "",
  ICS_URL_BASE = "",
  OCR_API_KEY = "",
  PORT = 3000,
} = process.env;

const OPENAI_MODEL     = process.env.OPENAI_MODEL || "gpt-4o-mini";
const EMBEDDING_MODEL  = process.env.EMBEDDING_MODEL || "text-embedding-3-small";

// === OpenAI client ===
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// === Tiny utils ===
const two = s => (s || "").slice(0, 2).toLowerCase();
const clamp = (s, n) => (s || "").length > n ? (s || "").slice(0, n) : (s || "");
const isPDF = (buf, filename = "") =>
  filename.toLowerCase().endsWith(".pdf") ||
  (buf?.[0] === 0x25 && buf?.[1] === 0x50 && buf?.[2] === 0x44 && buf?.[3] === 0x46); // %PDF

// === WhatsApp helpers ===
async function sendWA(to, payload) {
  const url = `https://graph.facebook.com/v20.0/${PHONE_NUMBER_ID}/messages`;
  try {
    const r = await axios.post(
      url,
      { messaging_product: "whatsapp", to, ...payload },
      { headers: { Authorization: `Bearer ${WHATSAPP_TOKEN}`, "Content-Type": "application/json" } }
    );
    return r.data;
  } catch (e) {
    console.error("sendWA error:", e?.response?.data || e.message);
  }
}
const sendText = (to, body) => sendWA(to, { type: "text", text: { body, preview_url: false } });

async function getMediaUrl(id) {
  const r = await axios.get(`https://graph.facebook.com/v20.0/${id}`, {
    headers: { Authorization: `Bearer ${WHATSAPP_TOKEN}` },
  });
  return r.data?.url;
}
async function downloadMedia(url) {
  const r = await axios.get(url, {
    headers: { Authorization: `Bearer ${WHATSAPP_TOKEN}` },
    responseType: "arraybuffer",
  });
  return Buffer.from(r.data);
}

// === KB with Embeddings ===
const KB_DIR = path.join(__dirname, "kb");
let KB_DOCS = [];       // [{name, text}]
let KB_CHUNKS = [];     // [{id, doc, chunk, text, embedding: number[]}]
const CHUNK_SIZE = 1200;
const CHUNK_OVERLAP = 200;

function safeRead(p) { try { return fs.readFileSync(p, "utf8"); } catch { return ""; } }

function splitIntoChunks(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const out = [];
  let i = 0;
  while (i < text.length) {
    const end = Math.min(text.length, i + size);
    const slice = text.slice(i, end);
    out.push(slice.trim());
    if (end === text.length) break;
    i = end - overlap;
    if (i < 0) i = 0;
  }
  return out.filter(s => s.length > 0);
}

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

async function embedBatch(texts) {
  const resp = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts
  });
  return resp.data.map(d => d.embedding);
}

async function buildKBEmbeddings() {
  KB_DOCS = [];
  KB_CHUNKS = [];

  if (!fs.existsSync(KB_DIR)) {
    console.log("KB: /kb not found — skipping.");
    return;
  }

  const files = fs.readdirSync(KB_DIR).filter(f => /\.(md|txt)$/i.test(f));
  if (!files.length) {
    console.log("KB: no .md/.txt files.");
    return;
  }

  console.log(`KB: reading ${files.length} file(s)…`);
  for (const fn of files) {
    const text = safeRead(path.join(KB_DIR, fn));
    if (!text.trim()) continue;
    KB_DOCS.push({ name: fn, text });
  }

  const allChunks = [];
  for (const d of KB_DOCS) {
    const chunks = splitIntoChunks(d.text);
    chunks.forEach((c, idx) => {
      allChunks.push({ doc: d.name, chunk: idx, text: c });
    });
  }
  console.log(`KB: chunked into ${allChunks.length} chunk(s).`);

  const BATCH = 64;
  let idCounter = 0;
  for (let i = 0; i < allChunks.length; i += BATCH) {
    const batch = allChunks.slice(i, i + BATCH);
    const embs = await embedBatch(batch.map(b => b.text));
    embs.forEach((e, j) => {
      const b = batch[j];
      KB_CHUNKS.push({ id: idCounter++, doc: b.doc, chunk: b.chunk, text: b.text, embedding: e });
    });
    console.log(`KB: embedded ${Math.min(i + BATCH, allChunks.length)}/${allChunks.length}`);
  }
  console.log(`KB: embeddings ready (${KB_CHUNKS.length} chunks).`);
}

function topKRelevant(queryEmbedding, k = 6) {
  if (!KB_CHUNKS.length) return [];
  const scored = KB_CHUNKS.map(c => ({ c, score: cosineSim(queryEmbedding, c.embedding) }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).map(s => s.c);
}

function formatContextFromChunks(chunks, maxChars = 7000) {
  if (!chunks.length) return "";
  let acc = "### KB matched context\n";
  let used = 0;
  for (const ch of chunks) {
    const block = `\n----- ${ch.doc} [${ch.chunk}] -----\n${ch.text}\n`;
    if (used + block.length > maxChars) break;
    acc += block;
    used += block.length;
  }
  return acc;
}

// initial load
await buildKBEmbeddings().catch(err => console.error("KB build error:", err?.response?.data || err.message));

// === Language memory ===
const userLang = new Map(); // phone => 'fi' | 'ru' | 'en' | 'bn' | 'ne' | ...

async function detectLangByText(text) {
  try {
    const r = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      temperature: 0,
      messages: [{
        role: "user",
        content:
`Return ONLY the two-letter ISO 639-1 code of the language of this message (lowercase). If unsure, reply 'en'.
Message:
"""${clamp(text, 600)}"""`,
      }],
    });
    const code = (r.choices?.[0]?.message?.content || "en").trim().toLowerCase();
    return /^[a-z]{2}$/.test(code) ? code : "en";
  } catch { return "en"; }
}

async function ensureUserLang(from, valueObj, sampleText) {
  if (userLang.has(from)) return userLang.get(from);
  const sys =
    valueObj?.contacts?.[0]?.locale ||
    valueObj?.contacts?.[0]?.language || "";
  if (sys) {
    const code = two(sys);
    if (/^[a-z]{2}$/.test(code)) { userLang.set(from, code); return code; }
  }
  const guess = await detectLangByText(sampleText || "");
  userLang.set(from, guess);
  return guess;
}

// Translate UI/system phrases to user's lang (fallback EN)
// (обновили system-промпт: никаких «извините/ограничений»)
async function trFor(user, english) {
  const lang = userLang.get(user) || "en";
  if (lang === "en") return english;
  try {
    const r = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      temperature: 0.2,
      messages: [
        { role: "system",
          content: `Translate the following UI string to ${lang}.
Keep it short and natural. Do not add apologies, disclaimers, or capability statements.` },
        { role: "user", content: english },
      ],
    });
    return r.choices?.[0]?.message?.content?.trim() || english;
  } catch {
    return english;
  }
}

// === Chat using retrieved KB chunks (обновили system-промпт) ===
async function chatWithKB(userText, userLangCode="en") {
  let ctx = "";
  try {
    const q = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: userText
    });
    const qEmb = q.data[0].embedding;
    const top = topKRelevant(qEmb, 6);
    ctx = formatContextFromChunks(top, 7000);
  } catch (e) {
    console.error("embed query error:", e?.response?.data || e.message);
  }

  const system =
`You are SOL — a warm, human assistant for SOL employees in Finland.
- Respond in the user's current language (${userLangCode}). If the user writes in another language, follow the user's latest message language.
- Be concise (3–7 short sentences), friendly, and practical.
- Prefer ONLY facts from [KB CONTEXT] for SOL rules/rights/chemicals/safety. If the answer is not in [KB CONTEXT], say you don't know and suggest checking with a supervisor/HR.
- Do not mention training data, knowledge cutoffs, or your internal limitations unless explicitly asked.
- Do not say you can answer only in one language. Just answer in the language the user is using.`;

  const messages = [
    { role: "system", content: system + (ctx ? `\n\n[KB CONTEXT]\n${ctx}` : "") },
    { role: "user", content: userText }
  ];

  try {
    const r = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      temperature: 0.25,
      messages
    });
    return r.choices?.[0]?.message?.content?.trim() || "OK.";
  } catch (e) {
    console.error("chatWithKB error:", e?.response?.data || e.message);
    return "OK.";
  }
}

// === OCR — OpenAI Vision (primary) + OCR.Space fallback ===
async function ocrImageBuffer(buf) {
  try {
    const dataUrl = `data:image/jpeg;base64,${buf.toString("base64")}`;
    const resp = await openai.responses.create({
      model: OPENAI_MODEL,
      temperature: 0,
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "Extract ONLY the raw text from this image. Keep line breaks. No commentary." },
            { type: "input_image", image_url: dataUrl }
          ]
        }
      ]
    });
    const out = resp.output_text?.trim?.() || "";
    if (out) return out;
  } catch (e) {
    console.error("OpenAI Vision OCR error:", e?.response?.data || e.message);
  }

  if (!OCR_API_KEY) return "";
  try {
    const form = new FormData();
    form.append("base64Image", `data:image/jpeg;base64,${buf.toString("base64")}`);
    form.append("language", "eng,ben,hin,nep,sin,tam,urd,ara,fre,spa,por,tha,rus,fin");
    form.append("isTable", "true");
    form.append("OCREngine", "2");
    const r = await axios.post("https://api.ocr.space/parse/image", form, {
      headers: { apikey: OCR_API_KEY, ...form.getHeaders() },
      maxBodyLength: Infinity
    });
    const parsed = r.data?.ParsedResults?.[0]?.ParsedText || "";
    return parsed.trim();
  } catch (e2) {
    console.error("OCR.Space error:", e2?.response?.data || e2.message);
    return "";
  }
}

// === PAM pay table helper ===
function currentPamTable(date = new Date()) {
  const d = new Date(date);
  if (d >= new Date('2027-07-01')) return '2027';
  if (d >= new Date('2026-08-01')) return '2026';
  return '2025';
}

const PAM_HOURLY = {
  "2025": [11.03,12.26,12.88,13.52,14.20,14.90,15.50,16.12,16.77,17.44],
  "2026": [11.33,12.59,13.22,13.88,14.58,15.30,15.92,16.55,17.22,17.90],
  "2027": [11.60,12.89,13.54,14.21,14.92,15.67,16.30,16.95,17.63,18.33],
};

function getHourlyByGroup(group1to10, date = new Date()) {
  const yearKey = currentPamTable(date);
  const arr = PAM_HOURLY[yearKey];
  const idx = Math.max(1, Math.min(10, group1to10)) - 1;
  return { rate: arr[idx], table: yearKey };
}

function groupFromPoints(points) {
  if (points < 17) return 1;
  if (points <= 20) return 2;
  if (points <= 24) return 3;
  if (points <= 28) return 4;
  if (points <= 33) return 5;
  if (points <= 38) return 6;
  if (points <= 44) return 7;
  if (points <= 51) return 8;
  if (points <= 58) return 9;
  return 10;
}


// === Welcome (once per number) ===
const seenUsers = new Set();
async function maybeSendWelcome(from) {
  if (seenUsers.has(from)) return;
  seenUsers.add(from);
  const msgEN = [
    "Hi! I’m SOL — your friendly assistant.",
    "I can:",
    "• Answer questions about working at SOL (rights, policies, safety, chemicals).",
    "• Read & translate screenshots/photos.",
    "• Help with cleaning techniques.",
    "• Share shift schedule links — just ask (e.g. “send my schedule”).",
  ].join("\n");
  await sendText(from, await trFor(from, msgEN));
}



function parseArrowTranslate(m) {
  const t = m.match(/^->\s*([a-z]{2})\s+([\s\S]+)$/i);
  if (!t) return null;
  return { target: t[1].toLowerCase(), text: m.slice(t[0].indexOf(t[2])).trim() || t[2].trim() };
}

// === Universal "schedule" intent detector (hybrid) ===

// 1) быстрый словарик
const SCHEDULE_KEYWORDS = [
  "schedule","shift","calendar","horario","calendario","grafik","duty",
  "расписание","смен","календарь",
  "aikataulu","vuorolista",
  "समय","कार्यतालिका","शिफ्ट",
  "সময়সূচি","ক্যালেন্ডার","শিফট",
  "时间表","班表","工作时间",
  "勤務表","シフト","スケジュール"
];
const SCHEDULE_COMBOS = [
  ["today","shift"], ["work","hours"], ["job","time"],
  ["сегодня","смен"], ["график","работ"], ["vuoro","tänään"],
  ["আজ","শিফট"], ["班","今天"]
];

function fastScheduleHit(text) {
  const t = (text || "").toLowerCase();
  if (SCHEDULE_KEYWORDS.some(w => t.includes(w))) return true;
  for (const [a,b] of SCHEDULE_COMBOS) {
    if (t.includes(a) && t.includes(b)) return true;
  }
  return false;
}

// 2) точный AI-классификатор (отвечает yes/no)
async function isScheduleIntentAI(text, langCode) {
  try {
    const r = await axios.post("https://api.openai.com/v1/chat/completions", {
      model: process.env.OPENAI_CLASSIFIER_MODEL || "gpt-4o-mini",
      temperature: 0,
      messages: [
        {
          role: "system",
          content:
`You are a strict intent classifier. Answer exactly "yes" or "no".
Task: Does the user ask for shift schedule or a link to the schedule/calendar?`
        },
        {
          role: "user",
          content:
`Language: ${langCode || "unknown"}
Text: """${(text||"").slice(0,600)}"""` // защита от длинных сообщений
        }
      ]
    }, {
      headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` }
    });

    const out = r.data?.choices?.[0]?.message?.content?.trim().toLowerCase() || "no";
    return out.startsWith("y"); // yes -> true, иначе false
  } catch (e) {
    console.error("schedule intent AI error:", e?.response?.data || e.message);
    return false;
  }
}

// 3) общий вход
async function looksLikeScheduleRequestSmart(text, langCode) {
  if (fastScheduleHit(text)) return true;
  return await isScheduleIntentAI(text, langCode);
}

// === Handlers ===
async function handleIncomingText(from, valueObj, body) {
  const lang = await ensureUserLang(from, valueObj, body);
  await maybeSendWelcome(from);

  const m = (body || "").trim();

  // авто-переключение языка, если пользователь сменил язык в этом сообщении
  try {
    const latestCode = await detectLangByText(m);
    const prevCode = userLang.get(from);
    if (latestCode && latestCode !== prevCode) {
      userLang.set(from, latestCode);
      console.log(`Language switched for ${from}: ${prevCode} -> ${latestCode}`);
    }
  } catch {}

 if (await looksLikeScheduleRequestSmart(m, userLang)) {
  const parts = [
    `• ${await trFor(from, "Open page")}: ${INDEX_URL}`,
   // `• ${await trFor(from, "ICS base (choose your card)")}: ${ICS_URL_BASE}`
  ];
  await sendText(from, (await trFor(from, "Schedule")) + ":\n" + parts.join("\n"));
  return;
}
  

  // KB admin
  if (/^kb\??$/i.test(m)) {
    const list = KB_DOCS.length ? KB_DOCS.map(d => `• ${d.name}`).join("\n") : "(empty)";
    await sendText(from, `KB docs: ${KB_DOCS.length}\n${list}`);
    return;
  }
  if (/^kb:\s*reload$/i.test(m)) {
    await buildKBEmbeddings().catch(err => console.error("KB rebuild error:", err?.response?.data || err.message));
    await sendText(from, `KB reloaded: ${KB_DOCS.length} file(s), ${KB_CHUNKS.length} chunks.`);
    return;
  }



  // translate arrows
  const arrow = parseArrowTranslate(m);
  if (arrow) {
    try {
      const r = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0.15,
        messages: [
          { role: "system", content: `Translate to ${arrow.target}. Keep meaning, tone, and formatting.` },
          { role: "user", content: arrow.text }
        ]
      });
      const out = r.choices?.[0]?.message?.content?.trim() || arrow.text;
      await sendText(from, out);
    } catch (e) {
      console.error("translate error:", e?.response?.data || e.message);
      await sendText(from, await trFor(from, "Sorry, I couldn't translate right now."));
    }
    return;
  }

  // general Q&A — use retrieved KB chunks
  const answer = await chatWithKB(m, userLang.get(from) || lang || "en");
  await sendText(from, answer);
}

async function handleIncomingImage(from, mediaId, caption, valueObj) {
  const lang = await ensureUserLang(from, valueObj, caption || "");
  await maybeSendWelcome(from);

  // авто-переключение языка по подписи к изображению
  if (caption) {
    try {
      const latestCode = await detectLangByText(caption);
      const prevCode = userLang.get(from);
      if (latestCode && latestCode !== prevCode) {
        userLang.set(from, latestCode);
        console.log(`Language switched (image caption) for ${from}: ${prevCode} -> ${latestCode}`);
      }
    } catch {}
  }

  try {
    const url = await getMediaUrl(mediaId);
    const buf = await downloadMedia(url);

    if (isPDF(buf, "")) {
      await sendText(from, await trFor(from, "PDF reading is limited now. Please send a screenshot of the needed page."));
      return;
    }

    const text = await ocrImageBuffer(buf);
    if (!text) {
      await sendText(from, await trFor(from,
        "I couldn’t read text from the image. Please try a sharper photo or a clear screenshot."
      ));
      return;
    }

    await sendText(from, await trFor(from, "I read the text from your image. Here is the beginning:"));
    await sendText(from, text.slice(0, 900));

    if (caption && caption.trim()) {
      const q = `${caption}\n\n(Consider this OCR context):\n${text}`;
      const follow = await chatWithKB(q, userLang.get(from) || lang || "en");
      await sendText(from, follow);
    } else {
      await sendText(from, await trFor(from,
        "You can:\n• ask about this content\n• or translate it via '->xx ...' (e.g. '->fi <text>')."
      ));
    }
  } catch (e) {
    console.error("handleIncomingImage error:", e?.response?.data || e.message);
    await sendText(from, await trFor(from, "Sorry, image processing failed."));
  }
}

async function handleIncomingDocument(from, mediaId, filename, valueObj) {
  const lang = await ensureUserLang(from, valueObj, filename || "");
  await maybeSendWelcome(from);

  // (под документ автосмена языка обычно не нужна, но не помешает по названию)
  if (filename) {
    try {
      const latestCode = await detectLangByText(filename);
      const prevCode = userLang.get(from);
      if (latestCode && latestCode !== prevCode) {
        userLang.set(from, latestCode);
        console.log(`Language switched (doc filename) for ${from}: ${prevCode} -> ${latestCode}`);
      }
    } catch {}
  }

  try {
    const url = await getMediaUrl(mediaId);
    const buf = await downloadMedia(url);

    if (isPDF(buf, filename)) {
      await sendText(from, await trFor(from,
        "PDF reading is limited now. Please send a screenshot or an image of the needed page."
      ));
      return;
    }

    const text = await ocrImageBuffer(buf);
    if (!text) {
      await sendText(from, await trFor(from, "I couldn’t extract text. Try sending it as an image or screenshot."));
      return;
    }
    await sendText(from, await trFor(from, `I read the document${filename ? ` (${filename})` : ""}. Here is the beginning:`));
    await sendText(from, text.slice(0, 900));
  } catch (e) {
    console.error("handleIncomingDocument error:", e?.response?.data || e.message);
    await sendText(from, await trFor(from, "Sorry, document processing failed."));
  }
}

// === Webhook VERIFY (GET) ===
app.get("/webhook", (req, res) => {
  const mode = req.query["hub.mode"];
  const token = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];
  if (mode === "subscribe" && token === VERIFY_TOKEN) {
    console.log("Webhook verified.");
    return res.status(200).send(challenge);
  }
  return res.sendStatus(403);
});

// === Webhook RECEIVE (POST) ===
app.post("/webhook", async (req, res) => {
  try {
    const entry  = req.body?.entry?.[0];
    const change = entry?.changes?.[0];
    const value  = change?.value;
    const msg    = value?.messages?.[0];
    if (!msg) return res.sendStatus(200);

    const from = msg.from;
    const type = msg.type;

    if (type === "text") {
      await handleIncomingText(from, value, (msg.text?.body || "").trim());
    } else if (type === "image") {
      await handleIncomingImage(from, msg.image?.id, msg.image?.caption || "", value);
    } else if (type === "document") {
      await handleIncomingDocument(from, msg.document?.id, msg.document?.filename || "", value);
    } else {
      await sendText(from, await trFor(from,
        "I can handle text, images (screenshots) and simple documents. Type 'menu' for help."
      ));
    }
    return res.sendStatus(200);
  } catch (e) {
    console.error("webhook handler error:", e?.response?.data || e.message);
    return res.sendStatus(200);
  }
});

// === Health/Version ===
app.get("/", (_req, res) => res.send(`WhatsApp SOL assistant is running ✅ v${VERSION}`));
app.get("/version", (_req, res) => res.send(`SOL Assistant version ${VERSION}`));

app.listen(PORT, async () => {
  console.log(`Bot on port: ${PORT} (v${VERSION})`);
  try {
    const list = await openai.models.list();
    console.log("✅ OpenAI API ok:", list.data?.length ?? "n/a");
  } catch {
    console.log("⚠️ OpenAI check failed (but bot started).");
  }
});
