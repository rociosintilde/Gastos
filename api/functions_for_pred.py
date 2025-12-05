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
from categories import CATEGORIES
from rescatar_valor_numerico import levenshtein

# Build paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- Utilities ----

def preprocess_spanish_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

# -------------------- Helper Functions -------------------- #

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
    text_new, category, amount = separar_texto_valor(text)
    # Predict category
    
    # Save text + category to DB
    await save_text_to_db(text = text_new, category = category, chat_id=chat_id, amount = amount)
    
    # Send reply to Telegram
    await send_telegram_message(chat_id, f"{text}, transformado a {text_new} con categoría {category} y precio {amount}")
    
    return JSONResponse({"status": "text_received", "text": text, "category": category})

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