// server.js — SOL WhatsApp Assistant (Meta Webhook, multilingual, Vision OCR, KB + Embeddings)
// Version: 2025-11-09.r3 (fix salary + small talk; schedule bug; kb call)
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

const VERSION = "2025-11-09.r3";

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

// ==== Light memory & defaults ====
const DEFAULT_HOURLY = 12.26;  // базовая ставка по умолчанию (PAM группа 2)
const MAX_TURNS = 8;

const STATE = new Map(); // phone -> { lang, history: [{role,content}], profile: {hourly, hoursPerWeek, lastTopic} }

function getState(id) {
  if (!STATE.has(id)) STATE.set(id, { lang: 'en', history: [], profile: {} });
  return STATE.get(id);
}
function pushToHistory(id, role, content) {
  const s = getState(id);
  s.history.push({ role, content });
  while (s.history.length > MAX_TURNS) s.history.shift();
}
function setProfile(id, patch) {
  const s = getState(id);
  s.profile = { ...(s.profile||{}), ...patch };
  return s.profile;
}
function getProfile(id) {
  return getState(id).profile || {};
}

// ==== Salary helpers ====
// корректнее считать месяц как 52/12 ≈ 4.333 недели
function monthlyFromWeeklyHours(hourly, hoursPerWeek, weeksPerMonth = 52/12) {
  const h = Number(hoursPerWeek || 0);
  const r = Number(hourly || DEFAULT_HOURLY);
  return +(r * h * weeksPerMonth).toFixed(2); // €
}
function monthlyBy4Weeks(hourly, hoursPerWeek) {
  const h = Number(hoursPerWeek || 0);
  const r = Number(hourly || DEFAULT_HOURLY);
  return +(r * h * 4).toFixed(2); // €
}

// парсим «ставку» и «часы в неделю» из текста на разных языках (простые паттерны)
// ✅ Ставка: требуем явного указания валюты или слова "ставка",
// чтобы не путать с выражениями типа "15 часов".
function parseHourlyRate(text) {
  const t = (text || "").toLowerCase();
  const m =
    t.match(/ставк[аи]?:?\s*(\d{1,3}(?:[.,]\d{1,2})?)/i) ||
    t.match(/tunti?palkka[:\s]*?(\d{1,3}(?:[.,]\d{1,2})?)/i) ||
    t.match(/(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:€|eur)\s*\/?\s*(?:h|ч|hr)?\b/i) ||
    t.match(/€\s*(\d{1,3}(?:[.,]\d{1,2})?)\s*\/?\s*(?:h|ч|hr)?\b/i);
  if (!m) return null;
  const num = parseFloat(m[1].replace(",", "."));
  if (!isFinite(num) || num < 6 || num > 40) return null;
  return +(num.toFixed(2));
}

// ✅ Часы в неделю: поддерживаем варианты:
// - "25 ч/нед", "25 h/week", "25 tuntia viikossa"
// - "15 часов в день, 6 дней в неделю" → 15×6 = 90
function parseHoursPerWeek(text) {
  const t = (text || "").toLowerCase();

  // 1) прямое указание часов в неделю
  let m =
    t.match(/(\d{1,3})\s*(?:h|ч|t)\s*\/?\s*(?:week|нед|недел[юи]|vko|viikk)/i) ||
    t.match(/(\d{1,3})\s*(?:tunn(?:in|tia)?)\s*(?:viikossa|\/vko)/i);
  if (m) {
    const n = parseInt(m[1], 10);
    if (n > 0 && n <= 100) return n;
  }

  // 2) "X часов в день, Y дней в неделю" (ru/en/fi)
  const dayH =
    (t.match(/(\d{1,2})\s*час(?:а|ов)?\s*в\s*день/)?.[1]) ||
    (t.match(/(\d{1,2})\s*h(?:ours)?\s*per\s*day/)?.[1]) ||
    (t.match(/(\d{1,2})\s*tunn(?:in|tia)\s*päivässä/)?.[1]);
  const daysW =
    (t.match(/(\d{1,2})\s*дн(?:я|ей)?\s*в\s*недел[юи]/)?.[1]) ||
    (t.match(/(\d{1,2})\s*days?\s*per\s*week/)?.[1]) ||
    (t.match(/(\d{1,2})\s*päivää\s*viikossa/)?.[1]);

  if (dayH && daysW) {
    const n = parseInt(dayH, 10) * parseInt(daysW, 10);
    if (n > 0 && n <= 100) return n;
  }

  return null;
}
const SALARY_CALC_INTENT =
  /(посчитай|рассчита(?:й|ть)|сколько.*в\s*месяц|сколько.*получ[уи]|monthly|per\s*month|how\s*much\s*per\s*month)/i;
const SALARY_INTENT = SALARY_CALC_INTENT;

// === OpenAI client ===
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// ===== Translation helpers =====
const LANG_NAME_TO_CODE = {
  // ru names
  "русский": "ru", "английский": "en", "финский": "fi", "непальский": "ne",
  "бенгальский": "bn", "испанский": "es", "португальский": "pt", "французский": "fr",
  "немецкий": "de", "итальянский": "it", "украинский": "uk", "эстонский": "et",
  // fi names
  "suomi": "fi", "englanti": "en", "venäjä": "ru", "nepali": "ne", "bengali": "bn",
  // en names
  "russian": "ru", "english": "en", "finnish": "fi", "nepali": "ne", "bengali": "bn",
  "spanish": "es", "portuguese": "pt", "french": "fr", "german": "de", "italian": "it",
};

function langCodeFromName(word) {
  const w = (word || "").toLowerCase().trim();
  return LANG_NAME_TO_CODE[w] || ( /^[a-z]{2}$/.test(w) ? w : null );
}

// универсальный переводчик в код языка (не зависит от userLang)
async function translateTo(code, text) {
  if (!code || !text) return null;
  try {
    const r = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      temperature: 0.2,
      messages: [
        { role: "system", content: `Translate the user's message into language ${code}. Output only the translation.` },
        { role: "user", content: text }
      ]
    });
    return r.choices?.[0]?.message?.content?.trim() || null;
  } catch (e) {
    console.error("translateTo error:", e?.response?.data || e.message);
    return null;
  }
}

// Разбор команд перевода: "->fi ...", "переведи на финский ...", "käännä suomeksi ..."
function parseTranslateCommand(text) {
  const raw = (text || "").trim();
  if (!raw) return null;

  const lower = raw.toLowerCase();

  // 1) Формат: "->fi some text"
  let m = raw.match(/^->\s*([a-z]{2})\s+(.+)/i);
  if (m) {
    return {
      code: m[1].toLowerCase(),
      text: (m[2] || "").trim(),
    };
  }

  // --- карта языков по словам в команде ---
  const langMap = [
    { re: /(финск|suom)/, code: "fi" },        // финский
    { re: /(англ|english|engl)/, code: "en" }, // английский
    { re: /(русск|russ|venäjä)/, code: "ru" }, // русский
    { re: /(nepal|नेपाली)/, code: "ne" },      // непали
    { re: /(bengal|বাংলা|bengali)/, code: "bn" } // бенгали (на будущее)
  ];

  // 2) Русско-английский формат:
  // "переведи на финский ...", "перевод на английский: ..."
  m = lower.match(/^(переведи|перевод|перевести|translate)\s+(?:на|to)\s+([^\s:]+)\s*:?\s*(.*)$/);
  if (!m) {
    // 3) Финский формат: "käännä suomeksi ...", "käännä englanniksi ..."
    m = lower.match(/^(käännä)\s+([^\s:]+)\s*:?\s*(.*)$/);
  }

  if (m) {
    const langWord = m[2];        // "финский" / "suomeksi" / "englanniksi"
    const rest = m[3] || "";      // текст после команды (может быть пустым)
    let code = null;

    for (const lm of langMap) {
      if (lm.re.test(langWord)) {
        code = lm.code;
        break;
      }
    }

    if (!code) return null;

    return {
      code,
      text: rest.trim(),          // может быть пустой строкой — это ок
    };
  }

  return null;
}

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

// === Media utils ===
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
const userLang = new Map(); // phone => 'fi' | 'ru' | 'en' | ...
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

async function classifyIntentAI(text) {
  try {
    const r = await openai.chat.completions.create({
      model: OPENAI_MODEL, // у тебя уже есть
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content:
`You are an intent router. Return strict JSON with fields:
{"intent":"salary_calc|salary_info|schedule|translate|chitchat|other","confidence":0..1}
Rules:
- salary_calc: user asks to CALCULATE earnings (per month, per week, "how much if 25 h/week", "посчитай", numbers about hours/rate).
- salary_info: questions ABOUT wages/rates/policies without asking to calculate ("какая зарплата в SOL", "минимальная ставка", "palkka-asiat").
- schedule: shift/roster/calendar links/when do I work.
- translate: explicit translate command ("->fi", "переведи на ...").
- chitchat: small talk.
- other: anything else.` },
        { role: "user", content: text.slice(0, 1000) }
      ]
    });
    const json = JSON.parse(r.choices[0].message.content || "{}");
    const intent = json.intent || "other";
    const conf = Math.max(0, Math.min(1, Number(json.confidence || 0)));
    return { intent, confidence: conf };
  } catch (e) {
    console.error("classifyIntentAI error:", e?.response?.data || e.message);
    return { intent: "other", confidence: 0 };
  }
}

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

// === Chat using retrieved KB chunks ===
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
- Prefer ONLY facts from [KB CONTEXT] for SOL rules/rights/chemicals/safety. If the answer is not in [KB CONTEXT], say you don't know and suggest checking with a supervisor/HR.`;

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

// === Schedule intent detector (словари + AI) ===
const SCHEDULE_KEYWORDS = [
  "schedule","shift","calendar","horario","calendario","grafik","duty",
  "расписание","смен","календарь",
  "aikataulu","vuorolista",
  "সময়সূচি","ক্যালেন্ডার","শিফট",
  "समय","कार्यतालिका","शिफ्ट",
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
async function isScheduleIntentAI(text, langCode) {
  try {
    const r = await axios.post("https://api.openai.com/v1/chat/completions", {
      model: process.env.OPENAI_CLASSIFIER_MODEL || "gpt-4o-mini",
      temperature: 0,
      messages: [
        { role: "system", content: `You are a strict intent classifier. Answer exactly "yes" or "no". Task: Does the user ask for shift schedule or a link to the schedule/calendar?` },
        { role: "user", content: `Language: ${langCode || "unknown"}\nText: """${(text||"").slice(0,600)}"""` }
      ]
    }, { headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` }});
    const out = r.data?.choices?.[0]?.message?.content?.trim().toLowerCase() || "no";
    return out.startsWith("y");
  } catch (e) {
    console.error("schedule intent AI error:", e?.response?.data || e.message);
    return false;
  }
}

// Небольшой помощник для вопросов в KB с учётом контекста
function buildKbQuery(message, st = {}) {
  const m = (message || "").trim();

  // Если короткое уточнение по предыдущему вопросу —
  // добавляем его в запрос, чтобы KB понимала контекст.
  if (m.length <= 40 && st.lastKbQuestion) {
    return `Previous user question: "${st.lastKbQuestion}".\nUser clarifies: "${m}".`;
  }

  // Обычный полный вопрос — отправляем как есть.
  return m;
}

async function looksLikeScheduleRequestSmart(text, langCode) {
  if (fastScheduleHit(text)) return true;
  return await isScheduleIntentAI(text, langCode);
}

// === User state (ставка/часы) + small talk ===
const USER_STATE = new Map(); // phone -> { rate?: number, hoursPerWeek?: number }
const CHITCHAT_RE =
  /^(привет|hi|hello|hei|moikka|hola|salut|как дела\??|что нового\??|yo)$/i;





// === Handlers ===
async function handleIncomingText(from, valueObj, body) {
  const lang = await ensureUserLang(from, valueObj, body);
  const m = (body || "").trim();

  // 1) Состояние пользователя (память) + история для перевода
  const st = USER_STATE.get(from) || {};
  if (st.lastText) st.lastTextPrev = st.lastText;
  st.lastText = m;
  USER_STATE.set(from, st);

  // 2) Парсим ставку и часы СРАЗУ
  const foundRate  = parseHourlyRate(m);
  const foundHours = parseHoursPerWeek(m);

  if (st.rate && st.rate < 7) delete st.rate; // защитимся от мусора
  if (typeof foundRate === "number") st.rate = foundRate;
  if (typeof foundHours === "number") st.hoursPerWeek = foundHours;
  if (foundRate || foundHours) USER_STATE.set(from, st);

  // 3) Сброс ставки/часов
  if (/^(reset|сброс)\s*(rate|ставка)?/i.test(m)) {
    USER_STATE.delete(from);
    await sendText(
      from,
      lang === "ru"
        ? "Ок, сбросил ставку и часы для расчетов."
        : lang === "fi"
        ? "Ok, nollasin tuntipalkan ja viikkotunnit."
        : "Okay, I reset hourly rate and weekly hours."
    );
    return;
  }

  // 4) Просто поболтать (короткие приветствия)
  if (CHITCHAT_RE.test(m) && m.length <= 60) {
    await sendText(
      from,
      lang === "ru"
        ? "Конечно, можем просто поболтать 😊 Как ты сегодня?"
        : lang === "fi"
        ? "Totta kai, voidaan vain jutella 😊 Miten päiväsi on mennyt?"
        : "Sure, we can just chat 😊 How’s your day going?"
    );
    return;
  }

  // 5) Перевод: "переведи на финский ...", "->fi ...".
  // Если текста нет — переведём предыдущее сообщение пользователя.
  const trCmd = parseTranslateCommand(m);
  if (trCmd && trCmd.code) {
    const sourceText =
      trCmd.text && trCmd.text.length > 0
        ? trCmd.text
        : (st.lastTextPrev || st.lastText || "");

    if (!sourceText) {
      await sendText(
        from,
        await trFor(
          from,
          "Send the text to translate (or write like: ->fi your text)."
        )
      );
      return;
    }

    const translated = await translateTo(trCmd.code, sourceText);
    if (translated) {
      await sendText(from, translated);
    } else {
      await sendText(
        from,
        await trFor(from, "Sorry, I couldn’t translate this.")
      );
    }
    return;
  }

  // 6) Авто-переключение языка по последнему сообщению
  try {
    const latestCode = await detectLangByText(m);
    const prevCode = userLang.get(from);
    if (latestCode && latestCode !== prevCode) {
      userLang.set(from, latestCode);
      console.log(`Language switched for ${from}: ${prevCode} -> ${latestCode}`);
    }
  } catch {}

  // 7) Расписание — умный детектор (словари + AI)
  if (await looksLikeScheduleRequestSmart(m, lang)) {
    await sendText(from, `${await trFor(from, "Schedule")}: ${INDEX_URL}`);
    return;
  }

  // 8) Детерминированный расчёт зарплаты
  const wantsSalary =
    SALARY_INTENT.test(m) ||
    typeof foundHours === "number" ||
    typeof foundRate === "number";

  if (wantsSalary) {
    const rate =
      typeof foundRate === "number"
        ? foundRate
        : st.rate ?? DEFAULT_HOURLY;

    const hours =
      typeof foundHours === "number"
        ? foundHours
        : st.hoursPerWeek;

    if (!hours) {
      await sendText(
        from,
        await trFor(
          from,
          `Tell me your weekly hours. I’ll use €${rate.toFixed(
            2
          )}/h by default.`
        )
      );
      return;
    }

    const by433 = monthlyFromWeeklyHours(rate, hours, 52 / 12);
    const by4   = monthlyBy4Weeks(rate, hours);

    let replyBase =
      `Here’s the estimate based on your data:\n` +
      `• Hourly rate: €${rate.toFixed(2)}/h\n` +
      `• Hours per week: ${hours}\n\n` +
      `Approximate monthly pay:\n` +
      `• Using 52/12 (≈4.33 weeks): €${by433}\n` +
      `• Using 4 weeks: €${by4}\n\n` +
      `Pay is based on actual hours worked. I can recalculate anytime.\n\n` +
      `💬 These amounts are before taxes.`;

    const replyTranslated = await trFor(from, replyBase);
    await sendText(from, replyTranslated);
    return;
  }

  // 9) Вопрос ПРО зарплату (инфо) — идём в KB, но с учётом контекста
  if (/(зарплат|ставк|palkka|rate|salary)/i.test(m)) {
    const kbQuestion = buildKbQuery(m, st);
    const kbAnswer = await chatWithKB(
      kbQuestion,
      userLang.get(from) || lang || "en"
    );
    st.lastKbQuestion = kbQuestion;
    USER_STATE.set(from, st);
    await sendText(from, kbAnswer);
    return;
  }

  // 10) Остальное — универсальный ассистент на базе KB, тоже с контекстом
  const kbQuestion = buildKbQuery(m, st);
  const follow = await chatWithKB(
    kbQuestion,
    userLang.get(from) || lang || "en"
  );
  st.lastKbQuestion = kbQuestion;
  USER_STATE.set(from, st);
  await sendText(from, follow);
}


async function handleIncomingImage(from, mediaId, caption, valueObj) {
  const lang = await ensureUserLang(from, valueObj, caption || "");

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
