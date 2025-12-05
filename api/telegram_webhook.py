import os
import logging
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import sys


sys.path.append(os.path.dirname(__file__))
from functions_for_pred import process_text_message, format_summaries_as_table, modify_last_purchase_cat

app = FastAPI()

# Logging
logging.basicConfig(
    level=logging.INFO,HF_TOKEN     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.post("/telegram_webhook")
async def telegram_webhook(req: Request):
    try:
        data = await req.json()
        logger.info(f"Received webhook data: {data}")
        message = data.get("message")
        if not message:
            return JSONResponse({"status": "no_message"})

        if message.get("text"):
            
            splitted = message["text"].split()
        
            if splitted and splitted[0] == "Cor":
                if len(splitted) > 1:
                    return await modify_last_purchase_cat(splitted[1], message["chat"]["id"])
                else:
                    # handle missing argument
                    return "Missing category parameter."
        
            elif message["text"] == "Reporte":
                return await format_summaries_as_table(message["chat"]["id"])
        
            else:
                return await process_text_message(message["text"], message["chat"]["id"])
    
        else:
            logger.warning(f"Unknown message type: {message.keys()}")
            return JSONResponse({"status": "unknown_message_type"})

    except Exception as e:
        logger.error(f"Unhandled error in webhook: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@app.get("/favicon.ico")
async def faviconico():
    return Response(status_code=204)


@app.get("/favicon.png")
async def faviconpng():
    return Response(status_code=204)


@app.get("/")
def read_root():
    return {"message": "Hello World from FastAPI on Vercel!"}


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/test-hf")
async def test_hf():
    try:
        logger.info("Testing Hugging Face client")
        return {"status": "hf_client_initialized", "provider": "fal-ai"}
    except Exception as e:
        logger.error(f"HF client test failed: {str(e)}")
        return {"status": "error", "error": str(e)}