import express from "express";
import axios from "axios";
import OpenAI from "openai";

const app = express();
app.use(express.json({ limit: "10mb" }));

// ===== CONFIG =====
const VERIFY_TOKEN = process.env.WHATSAPP_VERIFY_TOKEN;
const WHATSAPP_TOKEN = process.env.WHATSAPP_TOKEN;
const WHATSAPP_PHONE_ID = process.env.WHATSAPP_PHONE_ID;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const KB_FILES = "./kb"; // папка с файлами SOL (мы подключим позже)

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
// Модель для всех вызовов ИИ
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";

// ===== HELPERS =====

// отправка текста в WhatsApp
async function sendText(to, text) {
  try {
    await axios({
      method: "POST",
      url: "https://graph.facebook.com/v19.0/" + process.env.WHATSAPP_PHONE_ID + "/messages",
      headers: {
        Authorization: "Bearer " + WHATSAPP_TOKEN,
        "Content-Type": "application/json",
      },
      data: {
        messaging_product: "whatsapp",
        to,
        type: "text",
        text: { body: text },
      },
    });
  } catch (err) {
    console.error("sendText error:", err?.response?.data || err.message);
  }
}

// ====== WEBHOOK VERIFY ======
app.get("/webhook", (req, res) => {
  const mode = req.query["hub.mode"];
  const token = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];

  if (mode === "subscribe" && token === VERIFY_TOKEN) {
    return res.status(200).send(challenge);
  }
  return res.sendStatus(403);
});

// ====== WEBHOOK HANDLER ======
app.post("/webhook", async (req, res) => {
  try {
    const entry = req.body?.entry?.[0];
    const change = entry?.changes?.[0];
    const msg = change?.value?.messages?.[0];

    if (!msg) return res.sendStatus(200);

    const from = msg.from;
    let text = msg.text?.body || "";

    console.log("Incoming:", from, text);

    handleIncoming(from, text); // логика будет ниже

    res.sendStatus(200);
  } catch (err) {
    console.error("Webhook error:", err);
    res.sendStatus(500);
  }
});

// ===== USER STATE (простая память по номеру) =====
const userState = new Map(); // phone -> { lastUserText?: string, lastIntent?: string }

// ===== ИИ-маршрутизатор =====
// Определяет, что хочет пользователь: перевод, болтовню или вопрос к "умному ассистенту"
async function classifyMessageAI(message, st = {}) {
  const prompt = `
Ты — маршрутизатор для ассистента SOL в WhatsApp.

Определи, что хочет пользователь, и верни JSON БЕЗ лишнего текста.
Разрешённые intent:
- "translation"   — перевод текста
- "chitchat"      — просто поболтать
- "kb"            — вопрос к базе знаний SOL
- "salary_calc"   — расчёт зарплаты
- "schedule"      — запрос расписания

Всегда определи:
- "user_language" — основной язык пользователя (две буквы: "ru", "fi", "en" и т.п.).

Если intent = "translation", добавь:
- "target_language"       — язык перевода (две буквы: "fi", "en", "ru", "ne", "bn" и т.д.).
- "text_for_translation"  — что именно нужно перевести.
    * Если пользователь написал команду и тект в ОДНОМ сообщении:
      - в "text_for_translation" положи этот текст (без лишних объяснений).
    * Если пользователь написал только команду типа "переведи это на английский",
      а сам текст был В ПРЕДЫДУЩЕМ сообщении пользователя,
      поставь "text_for_translation": "" (пустая строка) — тогда бот возьмёт прошлое сообщение.

Верни СТРОГО один JSON-объект без пояснений, без комментариев, без Markdown.
Примеры корректных ответов:
{"intent":"translation","user_language":"ru","target_language":"fi","text_for_translation":"Здравствуйте, как дела?"}
{"intent":"chitchat","user_language":"ru"}
{"intent":"kb","user_language":"fi"}
{"intent":"salary_calc","user_language":"ru","hours_per_week":30,"hourly_rate":12.26}
{"intent":"schedule","user_language":"fi"}
`;

  const userPayload = {
    message,
    prev_intent: st.lastIntent || null,
    prev_user_text: st.lastUserText || null,
  };

  const resp = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    temperature: 0,
    messages: [
      { role: "system", content: prompt },
      {
        role: "user",
        content: JSON.stringify(userPayload, null, 2),
      },
    ],
  });

  let raw = resp.choices[0]?.message?.content || "{}";
  raw = raw.trim();

  try {
    const obj = JSON.parse(raw);
    if (!obj.intent) obj.intent = "kb";
    if (!obj.user_language) obj.user_language = "en";
    return obj;
  } catch (e) {
    console.error("Router JSON parse error:", e, raw);
    return { intent: "kb", user_language: "en" };
  }
}

// ===== Перевод через OpenAI =====
const LANG_NAMES = {
  ru: "Russian",
  fi: "Finnish",
  en: "English",
  ne: "Nepali",
  bn: "Bengali",
};

async function translateWithOpenAI(text, targetLang, sourceLang) {
  const langName = LANG_NAMES[targetLang] || targetLang || "English";

  const messages = [
    {
      role: "system",
      content:
        `You are a professional translator. ` +
        `Translate the USER text into ${langName}. ` +
        `Do not explain, do not add comments, return ONLY the translated text.`,
    },
    {
      role: "user",
      content: text,
    },
  ];

  const resp = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    temperature: 0,
    messages,
  });

  return resp.choices[0]?.message?.content?.trim() || "";
}

// ===== Общий "умный" ответ (без KB, пока просто ИИ) =====
async function smartAssistantReply(message, userLang) {
  const sys = `
Ты — дружелюбный ассистент для сотрудников SOL.
Отвечай кратко, по делу и на языке пользователя.
Если вопрос не про SOL (работа, химикаты, больничный, отпуск, графики, зарплата),
всё равно отвечай, но мягко и нейтрально.
`;

  const resp = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    temperature: 0.3,
    messages: [
      { role: "system", content: sys },
      { role: "user", content: message },
    ],
  });

  return resp.choices[0]?.message?.content?.trim() || "";
}

// ===== KB SOL (embeddings + поиск по md-файлам) =====
import fs from "fs";
import path from "path";

// Кэш embeddings, чтобы не пересчитывать при каждом запросе
const KB_CACHE = {
  files: [],
  embeddings: [],
  loaded: false,
};

// Читаем все md-файлы из папки kb
function loadKbFiles() {
  const kbDir = path.resolve(KB_FILES);
  const files = fs.readdirSync(kbDir).filter(f => f.endsWith(".md"));

  KB_CACHE.files = files.map(f => {
    const content = fs.readFileSync(path.join(kbDir, f), "utf8");
    return { name: f, content };
  });
}

// создаём embeddings для всех файлов
async function buildKbEmbeddings() {
  if (!KB_CACHE.files.length) loadKbFiles();

  const model = "text-embedding-3-small";

  KB_CACHE.embeddings = [];
  for (const file of KB_CACHE.files) {
    const resp = await openai.embeddings.create({
      model,
      input: file.content,
    });

    KB_CACHE.embeddings.push({
      name: file.name,
      embedding: resp.data[0].embedding,
      content: file.content,
    });
  }

  KB_CACHE.loaded = true;
  console.log("KB loaded:", KB_CACHE.embeddings.length, "files");
}

// косинусное расстояние
function similarity(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }

  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ищем самые релевантные документы
async function searchKb(query) {
  if (!KB_CACHE.loaded) {
    await buildKbEmbeddings();
  }

  // embedding запроса
  const embQ = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: query,
  });

  const q = embQ.data[0].embedding;

  // сортируем файлы по успешности
  const ranked = KB_CACHE.embeddings
    .map(doc => ({
      name: doc.name,
      score: similarity(q, doc.embedding),
      content: doc.content,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  return ranked;
}

// финальное сообщение на основе KB + модели
async function answerFromKb(query, userLang = "fi") {
  const top = await searchKb(query);

  const context = top
    .map(doc => `# File: ${doc.name}\n${doc.content}`)
    .join("\n\n");

  const prompt = `
Ты ассистент SOL. Используй информацию ТОЛЬКО из документов ниже.
Если точного ответа нет в документах — скажи это вежливо и мягко.

Ответ должен быть на языке пользователя (${userLang}).

=== DOCUMENTS ===
${context}

=== USER QUESTION ===
${query}

Ответь ясно, коротко и по делу.
`;

  const resp = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    temperature: 0.2,
    messages: [
      { role: "system", content: prompt },
    ],
  });

  return resp.choices[0]?.message?.content?.trim() || "";
}

// ===== Главный обработчик входящего текста =====
async function handleIncoming(from, text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return;

  const st = userState.get(from) || {};
  st.lastUserText = trimmed;

  // 1) Маршрутизатор просит ИИ решить, что делать
  let route;
  try {
    route = await classifyMessageAI(trimmed, st);
  } catch (err) {
    console.error("classifyMessageAI error:", err);
    route = { intent: "kb", user_language: "en" };
  }

  st.lastIntent = route.intent;
  userState.set(from, st);

    const userLang = route.user_language || "en";

  // 2) Обработка по intent
  if (route.intent === "translation") {
    let textToTranslate = (route.text_for_translation || "").trim();

    // если в маршрутизаторе текст пустой — берём предыдущее сообщение пользователя
    if (!textToTranslate) {
      textToTranslate = st.lastUserText || trimmed;
    }

    if (!textToTranslate) {
      await sendText(
        from,
        userLang === "ru"
          ? "Напишите, пожалуйста, текст, который нужно перевести."
          : "Please send the text you want me to translate."
      );
      return;
    }

    const translated = await translateWithOpenAI(
      textToTranslate,
      route.target_language || "en",
      userLang
    );

    if (!translated) {
      await sendText(
        from,
        userLang === "ru"
          ? "Не удалось перевести текст. Попробуйте сформулировать запрос чуть иначе."
          : "I couldn't translate that. Please try again with a slightly different request."
      );
      return;
    }

    await sendText(from, translated);
    return;
  }

  if (route.intent === "chitchat") {
    const reply = await smartAssistantReply(trimmed, userLang);
    await sendText(from, reply);
    return;
  }

    // ===== ЗАРПЛАТА (детерминированный расчёт) =====
  if (route.intent === "salary_calc") {
    const rate =
      typeof route.hourly_rate === "number" && route.hourly_rate > 6
        ? route.hourly_rate
        : 12.26; // ставка по умолчанию SOL

    const hours =
      typeof route.hours_per_week === "number" &&
      route.hours_per_week >= 5 &&
      route.hours_per_week <= 60
        ? route.hours_per_week
        : null;

    if (!hours) {
      await sendText(
        from,
        userLang === "ru"
          ? "Укажи, пожалуйста, сколько часов в неделю ты работаешь."
          : "Tell me your weekly working hours."
      );
      return;
    }

    // Формулы
    const by433 = (rate * hours * (52 / 12)).toFixed(2);
    const by4 = (rate * hours * 4).toFixed(2);

    let base = `
Hourly rate: €${rate.toFixed(2)}
Hours per week: ${hours}

Estimated monthly salary:
• 52/12 method (≈4.33 weeks): €${by433}
• 4-week method: €${by4}

💬 These amounts are BEFORE taxes.
`;

    const resp = await translateWithOpenAI(base, userLang);
    await sendText(from, resp);
    return;
  }

  // ===== РАСПИСАНИЕ (графики) =====
  if (route.intent === "schedule") {
    const url = process.env.INDEX_URL || "https://sol-droid-nik.github.io/Calendars/";
    let msg =
      userLang === "ru"
        ? `Твоё расписание: ${url}`
        : userLang === "fi"
        ? `Työvuorolistasi: ${url}`
        : `Your schedule: ${url}`;

    await sendText(from, msg);
    return;
  }
  

  if (route.intent === "kb") {
    const reply = await answerFromKb(trimmed, userLang);
    await sendText(from, reply);
    return;
  }

  // на всякий случай фоллбек — просто умный ответ
  const reply = await smartAssistantReply(trimmed, userLang);
  await sendText(from, reply);
}


// ===== START =====
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("V4 bot running on port", PORT));
