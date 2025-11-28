import numpy as np
import re
import os
import aiohttp
from fastapi.responses import JSONResponse
import asyncpg
from huggingface_hub import InferenceClient
from datetime import datetime, timedelta
import zoneinfo
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)
import sys
sys.path.append(os.path.dirname(__file__))
from rescatar_valor_numerico import separar_texto_valor

# Build paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model/mlp_embeddings_model.npz")
EMB_DIR = os.path.join(BASE_DIR, "model/combined_embeddings.vec")

# ---- Utilities ----

def preprocess_spanish_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def relu(x): return np.maximum(0, x)
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def load_local_spanish_embeddings(path=EMB_DIR):
    embeddings = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split(' ')
                if len(parts) < 2:
                    continue
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                embeddings[word] = vector
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return None


def create_embedding_matrix(embeddings_dict):
    vector_size = len(next(iter(embeddings_dict.values())))
    vocab_to_idx = {'<UNK>': 0}
    idx_to_vocab = {0: '<UNK>'}
    embedding_matrix = [np.random.normal(0, 0.1, vector_size)]
    for i, (word, vector) in enumerate(embeddings_dict.items(), 1):
        vocab_to_idx[word] = i
        idx_to_vocab[i] = word
        embedding_matrix.append(vector)
    return np.array(embedding_matrix), vocab_to_idx, idx_to_vocab, vector_size

def get_text_embedding(text, embedding_matrix, vocab_to_idx, vector_size):
    words = text.split()
    vectors = [embedding_matrix[vocab_to_idx.get(word, 0)] for word in words]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

# ---- Classifier ----

class SpanishTextClassifier:
    def __init__(self, model_path=MODEL_DIR, emb_path=EMB_DIR):
        # Load model weights
        data = np.load(model_path, allow_pickle=True)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.label_names = data["label_names"]

        # Load embeddings
        embeddings = load_local_spanish_embeddings(emb_path)
        self.embedding_matrix, self.vocab_to_idx, self.idx_to_vocab, self.vector_size = create_embedding_matrix(embeddings)

    def predict(self, text: str) -> str:
        text = preprocess_spanish_text(text)
        emb = get_text_embedding(text, self.embedding_matrix, self.vocab_to_idx, self.vector_size)
        x = emb.reshape(1, -1)
        h = relu(x @ self.W1 + self.b1)
        out = softmax(h @ self.W2 + self.b2)
        pred_idx = int(np.argmax(out, axis=1)[0])
        return self.label_names[pred_idx]

# ---- Singleton instance ----
_classifier_instance = None

def get_classifier() -> SpanishTextClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SpanishTextClassifier()
    return _classifier_instance

# ---- Function to keep your old interface ----
def predict_category(text: str) -> str:
    classifier = get_classifier()
    return classifier.predict(text)

# Environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN not found in environment variables")
if not HF_TOKEN:
    logger.error("HF_TOKEN not found in environment variables")

# Hugging Face client
client = InferenceClient(
    api_key=HF_TOKEN,
    headers={"Content-Type": "audio/ogg"})

# -------------------- Helper Functions -------------------- #

async def download_telegram_audio(file_id: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        file_info_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile?file_id={file_id}"
        async with session.get(file_info_url) as resp:
            if resp.status != 200:
                raise Exception(f"Telegram API returned status {resp.status}")
            file_info = await resp.json()
        if not file_info.get("ok"):
            raise Exception(f"Telegram API error: {file_info.get('description')}")
        file_path = file_info["result"]["file_path"]
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
        async with session.get(file_url) as audio_resp:
            if audio_resp.status != 200:
                raise Exception(f"Failed to download audio: {audio_resp.status}")
            return await audio_resp.read()


async def send_telegram_message(chat_id: int, text: str):
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )
        if response.status != 200:
            logger.error(f"Failed to send Telegram message: {response.status}")


async def save_text_to_db(text: str, category: str, chat_id, amount: int = 0):
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        tz = zoneinfo.ZoneInfo("America/Santiago")
        now = datetime.now(tz).replace(tzinfo=None)  # naive, Chilean local time
        try:
            await conn.execute(
                "INSERT INTO gastos_db (timestamp, gasto, tipo_de_gasto, monto) VALUES ($1, $2, $3, $4)",
                now, text, category, amount
            )
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"Database error: {str(e)}")

CATEGORIES = [
    "Alcohol",
    "Ocio y entretenimiento",
    "Restaurante y bares",
    "Salud y cuidado personal",
    "Supermercado",
    "Tonteras calle",
    "Transporte",
    "UBER delivery",
    "Vivienda y servicios",
]

def levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    m, n = len(a), len(b)
    if m > n:
        a, b = b, a
        m, n = n, m

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for j in range(1, n + 1):
        curr[0] = j
        for i in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(
                prev[i] + 1,
                curr[i - 1] + 1,
                prev[i - 1] + cost
            )
        prev, curr = curr, prev
    return prev[m]
async def modify_last_purchase_cat(text: str, chat_id: int):
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            lower_text = text.lower()

            # 1. Check prefix matches
            prefix_matches = [
                c for c in CATEGORIES
                if c.lower().startswith(lower_text)
            ]

            if len(prefix_matches) == 1:
                best_cat = prefix_matches[0]
            else:
                # 2. Fallback: closest by Levenshtein distance
                best_cat = min(CATEGORIES, key=lambda c: levenshtein(text, c))

            await conn.execute(
                """
                UPDATE gastos_db
                SET tipo_de_gasto = $1
                WHERE id = (
                    SELECT id
                    FROM gastos_db
                    ORDER BY timestamp DESC
                    LIMIT 1
                );
                """,
                best_cat,
            )

            # Send reply to Telegram
            await send_telegram_message(chat_id, f"last entry modified to categoría {best_cat}")
            
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"Database error: {str(e)}")

        
async def process_text_message(text: str, chat_id: int):
    text_new, amount = separar_texto_valor(text)
    # Predict category
    category = predict_category(text_new)
    
    # Save text + category to DB
    await save_text_to_db(text = text_new, category = category, chat_id=chat_id, amount = amount)
    
    # Send reply to Telegram
    await send_telegram_message(chat_id, f"{text}, transformado a {text_new} con categoría {category} y precio {amount}")
    
    return JSONResponse({"status": "text_received", "text": text, "category": category})

async def process_voice_message(message: dict):
    file_id = message["voice"]["file_id"]
    chat_id = message["chat"]["id"]
    
    # Transcribe audio
    try:
        ogg_bytes = await download_telegram_audio(file_id)
        logger.info(f"Downloaded {len(ogg_bytes)} bytes of audio")
        output = client.automatic_speech_recognition(
            ogg_bytes,
            model="openai/whisper-large-v3-turbo"
        )
        transcription = output.get('text', '')
        logger.info(f"Transcription: {transcription}")
    except Exception as e:
        await send_telegram_message(chat_id, f"Error processing audio: {str(e)}")
        return JSONResponse({"status": "error", "error": "speech_recognition_failed"})

    # Predict category and Save transcription + category to DB
    return await process_text_message(transcription, chat_id)

# -------------------- Reportería -------------------- #

async def fetch_expenses(chat_id: int):
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            rows = await conn.fetch(
                "SELECT timestamp, tipo_de_gasto, monto FROM gastos_db"
            )
            logger.error("fetch_expenses seemed to work")
            # Convert to list of dicts for easier processing
            return [
                {"timestamp": row["timestamp"], "tipo": row["tipo_de_gasto"], "monto": row["monto"]}
                for row in rows
            ]
        except Exception as e:
            await send_telegram_message(chat_id, f"didnt got the data from the ddbb {str(e)}")
        finally:
            await conn.close()
    except Exception as e:
        await send_telegram_message(chat_id, f"didnt connect to the ddbb {str(e)}")
        logger.error(f"Database error: {str(e)}")
        return []

def sum_by_category(expenses, start_date=None):
    """
    expenses: list of dicts with keys timestamp, tipo, monto
    start_date: optional datetime to filter by
    returns: dict {tipo_de_gasto: sum_of_monto}
    """
    sums = defaultdict(int)
    total = 0
    for e in expenses:
        if start_date and e["timestamp"] < start_date:
            continue
        sums[e["tipo"]] += e["monto"]
        total += e["monto"]
    sums["total"] = total
    
    return dict(sums)

def project_end_of_month(expenses):
    """
    Linear projection: current sum / days_passed * total_days_in_month
    """

    now = datetime.now()
    start_of_month = now.replace(day=1)
    # Days passed in month (including today)
    days_passed = (now - start_of_month).days + 1
    # Total days in month
    if now.month == 12:
        next_month = datetime(now.year + 1, 1, 1)
    else:
        next_month = datetime(now.year, now.month + 1, 1)
    total_days = (next_month - start_of_month).days
    
    sums_so_far = sum_by_category(expenses, start_date=start_of_month)
    
    projection = {tipo: monto / days_passed * total_days for tipo, monto in sums_so_far.items()}
    return projection

async def calculate_summaries(chat_id):
    
    expenses = await fetch_expenses(chat_id)
    
    now = datetime.now()
    # Last 7 days
    last_7_days = now - timedelta(days=7)
    
    # Last 31 days
    last_31_days = now - timedelta(days=31)
    
    # Start of week (Monday)
    start_of_week = now - timedelta(days=now.weekday())
    
    # Start of month
    start_of_month = now.replace(day=1)
    
    last7 = sum_by_category(expenses, last_7_days)
    last31 = sum_by_category(expenses, last_31_days)
    week = sum_by_category(expenses, start_of_week)
    month = sum_by_category(expenses, start_of_month)
    projection = project_end_of_month(expenses)
    
    return {
        "last_7_days": last7,
        "last_31_days": last31,
        "this_week": week,
        "this_month": month,
        "projection_end_of_month": projection,
    }

async def format_summaries_as_table(chat_id: int):

    summaries = await calculate_summaries(chat_id)

    msg = "*Expense Summary*\n\n"  # Markdown bold
    for period, data in summaries.items():
        msg += f"*{period.replace('_', ' ').title()}*\n"
        msg += "Tipo de Gasto | Monto\n"
        msg += "-------------|------\n"
        for tipo, monto in data.items():
            # Format monto as currency with $ and dot as thousands separator, no decimals
            formatted_monto = f"${int(monto):,}".replace(",", ".")
            msg += f"{tipo:<15} | {formatted_monto:>7}\n"
        msg += "\n"
    
    # Send reply to Telegram
    await send_telegram_message(chat_id, msg)
    
    return JSONResponse({"status": "returned report"})