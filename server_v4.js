import express from "express";
import axios from "axios";
import OpenAI from "openai";

const app = express();
app.use(express.json({ limit: "10mb" }));

// ===== CONFIG =====
const VERIFY_TOKEN = process.env.WHATSAPP_VERIFY_TOKEN;
const WHATSAPP_TOKEN = process.env.WHATSAPP_TOKEN;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const KB_FILES = "./kb"; // папка с файлами SOL (мы подключим позже)

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

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

// ===== START =====
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("V4 bot running on port", PORT));
