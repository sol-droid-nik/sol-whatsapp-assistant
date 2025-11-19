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
// Маршрутизатор: ИИ решает, что это за запрос и какие данные из него вытащить
async function classifyMessageAI(message, prevState = {}) {
  const text = (message || "").trim();

  // базовый объект по умолчанию
  const base = {
    intent: "kb",
    user_language: "ru",
    hours_per_week: null,
    hourly_rate: null,
    target_language: null,
    text_for_translation: null,
  };

  if (!text) return base;

  const systemPrompt = `
Ты — маршрутизатор для WhatsApp-бота компании SOL Palvelut.

Твоя задача — по одному сообщению пользователя:
1) Определить намерение (intent).
2) Определить язык пользователя (user_language).
3) При необходимости вытащить числовые параметры (ставка, часы).
4) При необходимости подготовить текст и язык для перевода.

Возможные intent:

- "translation" — пользователь просит ПЕРЕВЕСТИ текст
  Примеры:
    "Переведи на финский: Я завтра не приду"
    "->fi I am sick"
    "Can you translate this to English?"
    "Напиши это клиенту по-английски"

- "salary_calc" — пользователь просит ПОСЧИТАТЬ зарплату по ставке и часам.
  Примеры:
    "Посчитай зарплату при ставке 12,26 и 25 часов в неделю"
    "How much will I get per month if I work 30h/week with 12.26 €/h?"
    "Сколько я буду получать, если 20 часов в неделю по 12,26?"

  ВАЖНО:
  - salary_calc — только если пользователь ЯВНО просит посчитать
    и есть ставка/часы (или их можно разумно спросить).
  - Если он спрашивает просто "Какая зарплата в SOL", "Какие ставки по PAM",
    это НЕ расчёт, это информация → тогда intent НЕ "salary_calc".

- "kb" — информационный вопрос по внутренним правилам, PAM/TES, больничным,
  отпуску, химии, безопасности и т.п.
  Примеры:
    "Какая сейчас зарплата в SOL?"
    "Какие ставки по PAM для уборщиков?"
    "Как оплачивается больничный в SOL?"
    "Что говорит PAM про отпуск?"
    "Miten sairasloma maksetaan SOL:ssa?"
    "What does PAM say about salary in cleaning sector?"

  ОСОБЕННО:
  - Любые вопросы вида "какая зарплата / какие ставки / palkka / TES / PAM"
    БЕЗ просьбы СЧИТАТЬ месячную сумму → это intent "kb".
    Ответ должен идти из KB (PAM-файл и другие документы),
    а не через калькулятор.

- "schedule" — вопросы про расписание / рабочие смены / график.
  Примеры:
    "Скинь расписание"
    "Моё työvuorot"
    "My shifts link please"
    "Расписание на неделю"

- "chitchat" — просто поболтать, приветствие, small talk
  Примеры:
    "Привет", "Hei", "Hello", "Как дела?", "Miten menee?"

- "other" — всё остальное (общие вопросы, не связанные с SOL).
  Для "other" бот может отвечать как обычный ассистент.

user_language:
- "ru" — если преобладает русский
- "fi" — финский
- "en" — английский
- "ne" — непальский
- "bn" — бенгальский
- если не уверен — выбери тот, который больше всего подходит.

Если intent = "translation":
- target_language — язык, на который нужно перевести (например "fi", "en", "ru").
- text_for_translation — текст, который нужно перевести (если явно указан).

Если intent = "salary_calc":
- hours_per_week — число часов в неделю, если указаны (например 25).
- hourly_rate — ставка в евро в час, если указана (например 12.26).
- Если числа в сообщении — разделены запятой или точкой, приведи к числу.

Верни ТОЛЬКО JSON без пояснений в формате:
{
  "intent": "...",
  "user_language": "...",
  "hours_per_week": null,
  "hourly_rate": null,
  "target_language": null,
  "text_for_translation": null
}
`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4.1-mini",
      temperature: 0,
      messages: [
        { role: "system", content: systemPrompt },
        {
          role: "user",
          content: JSON.stringify({
            message: text,
            prev_intent: prevState.lastIntent || null,
            prev_language: prevState.user_language || null,
          }),
        },
      ],
    });

    let raw = completion.choices[0]?.message?.content?.trim() || "";
    // иногда модель может обернуть JSON в ```json ... ```
    raw = raw.replace(/```json/gi, "").replace(/```/g, "").trim();

    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch {
      // если не удалось распарсить — вернём базовый объект
      return base;
    }

    const result = {
      ...base,
      ...parsed,
    };

    // лёгкая нормализация
    if (typeof result.hours_per_week === "string") {
      const n = parseFloat(result.hours_per_week.replace(",", "."));
      result.hours_per_week = isFinite(n) ? n : null;
    }
    if (typeof result.hourly_rate === "string") {
      const n = parseFloat(result.hourly_rate.replace(",", "."));
      result.hourly_rate = isFinite(n) ? n : null;
    }

    if (typeof result.hours_per_week === "number") {
      if (result.hours_per_week <= 0 || result.hours_per_week > 80) {
        result.hours_per_week = null;
      }
    }

    if (typeof result.hourly_rate === "number") {
      if (result.hourly_rate < 6 || result.hourly_rate > 40) {
        result.hourly_rate = null;
      }
    }

    // нормализуем язык
    if (!result.user_language) result.user_language = "ru";

    return result;
  } catch (err) {
    console.error("classifyMessageAI error:", err);
    return base;
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
const KB_CHUNK_SIZE = 1500;       // размер кусочка в символах
const KB_CHUNK_OVERLAP = 200;     // перекрытие между кусками
// Помощник для формирования запроса в KB с учётом контекста
function buildKbQuery(message, st = {}) {
  const m = (message || "").trim();

  // Если пользователь написал короткое уточнение,
  // и у нас уже был предыдущий вопрос в KB —
  // отправим оба: старый + уточнение.
  if (m.length > 0 && m.length <= 60 && st.lastKbQuery) {
    return `Previous user question: "${st.lastKbQuery}".\nUser clarifies: "${m}".`;
  }

  // Иначе — просто берём текущий текст
  return m;
}


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
    const full = (file.content || "").trim();
    if (!full) continue;

    // режем файл на куски с overlap
    const chunks = [];
    for (
      let i = 0;
      i < full.length;
      i += KB_CHUNK_SIZE - KB_CHUNK_OVERLAP
    ) {
      const chunk = full.slice(i, i + KB_CHUNK_SIZE);
      if (chunk.trim()) {
        chunks.push(chunk);
      }
    }

    // если по какой-то причине ничего не получилось — пропускаем файл
    if (!chunks.length) continue;

    // одним запросом эмбеддим все чанки этого файла
    const resp = await openai.embeddings.create({
      model,
      input: chunks,
    });

    resp.data.forEach((item, idx) => {
      KB_CACHE.embeddings.push({
        name: file.name,
        embedding: item.embedding,
        content: chunks[idx],
      });
    });
  }

  KB_CACHE.loaded = true;
  console.log(
    "KB loaded chunks:",
    KB_CACHE.embeddings.length,
    "from files:",
    KB_CACHE.files.length
  );
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
  console.log("KB query:", query, "lang:", userLang);

  // 1) Попробуем загрузить и найти документы
  let top;
  try {
    top = await searchKb(query);
  } catch (e) {
    console.error("searchKb error:", e);
    // если совсем всё плохо с KB — пусть наверх уйдёт ошибка,
    // её перехватит handleIncoming и сделает fallback
    throw e;
  }

  if (!top || top.length === 0) {
    console.warn("KB: no documents found for query");
    // Честно скажем пользователю, что в KB ничего не нашли
    if (userLang === "ru") {
      return "Я посмотрел внутренние документы SOL, но не нашёл точной информации по этому вопросу. Попробуй переформулировать или спросить руководителя / HR.";
    }
    if (userLang === "fi") {
      return "Katsoin SOL:n sisäiset ohjeet, mutta en löytänyt tarkkaa vastausta. Voit kysyä esihenkilöltä tai HR:ltä.";
    }
    return "I checked the internal SOL documents but couldn’t find an exact answer. Please consider asking your supervisor or HR.";
  }

  // 2) Собираем контекст
  const context = top
    .map(doc => `# File: ${doc.name}\n${doc.content}`)
    .join("\n\n");

  console.log(
    "KB top docs:",
    top.map(d => ({ name: d.name, score: d.score }))
  );

  const prompt = `
Ты ассистент SOL. Используй информацию ТОЛЬКО из документов ниже.
Если точного ответа нет в документах — скажи это вежливо и мягко и порекомендуй спросить руководителя или HR.

Ответ должен быть на языке пользователя (${userLang}).

=== DOCUMENTS ===
${context}

=== USER QUESTION ===
${query}

Ответь ясно, коротко и по делу, ссылаясь только на то, что есть в документах.
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

    // если в маршрутизаторе текст пустой:
    // 1) пробуем перевести последний ответ бота
    // 2) иначе берём последнее сообщение пользователя
    if (!textToTranslate) {
      if (st.lastBotText) {
        textToTranslate = st.lastBotText;
      } else {
        textToTranslate = st.lastUserText || trimmed;
      }
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

    // запоминаем последний ответ бота
    st.lastBotText = translated;
    userState.set(from, st);

    await sendText(from, translated);
    return;
  }

    if (route.intent === "chitchat") {
    const reply = await smartAssistantReply(trimmed, userLang);
    st.lastBotText = reply;
    userState.set(from, st);
    await sendText(from, reply);
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
    st.lastBotText = msg;
    userState.set(from, st);
    await sendText(from, msg);
    return;
  }

  // ===== ЗАРПЛАТА (детерминированный расчёт) =====
if (route.intent === "salary_calc") {
  const rate =
    typeof route.hourly_rate === "number" && route.hourly_rate > 6
      ? route.hourly_rate
      : 12.26; // ставка по умолчанию

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
  st.lastBotText = resp;
  userState.set(from, st);
  await sendText(from, resp);
  return;
}

     if (route.intent === "kb") {
    // Сформируем запрос в KB с учётом контекста
    const kbQuery = buildKbQuery(trimmed, st);

    try {
      const reply = await answerFromKb(kbQuery, userLang);
      await sendText(from, reply);

      // запомним последний "основной" вопрос для будущих уточнений
      st.lastKbQuery = kbQuery;
      st.lastBotText = reply;
      userState.set(from, st);
    } catch (e) {
      console.error("answerFromKb error:", e);

      // запасной вариант — просто умный ассистент без KB,
      // чтобы бот не молчал
      const fallback = await smartAssistantReply(trimmed, userLang);
      st.lastBotText = fallback;
      userState.set(from, st);
      await sendText(from, fallback);
    }
    return;
  }

  // на всякий случай фоллбек — просто умный ответ
     const reply = await smartAssistantReply(trimmed, userLang);
     st.lastBotText = reply;
     userState.set(from, st);
     await sendText(from, reply);
}



// ===== START =====
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("V4 bot running on port", PORT));
