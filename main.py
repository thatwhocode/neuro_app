from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
from typing import List, Tuple
import json as js
import logging
import os 
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi.responses import JSONResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_PATH  = os.getenv("MODEL_PATH", "/app/model-best")
nlp: spacy.language.Language = None



class FabulaInput(BaseModel):
    fabula: str

class FabulaOutput(BaseModel):
    entities: List[Tuple[str, str]] = []



@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp
    try:
        logger.info(f"Спроа завантажити модель Spacy з : {MODEL_PATH} ")
        nlp = spacy.load(MODEL_PATH)
        logger.info("Модель успішно завантажена")
    except OSError:
        logger.exception(f"Не вдалося завантажити млдель Spacy з {MODEL_PATH}")
    except Exception as e :
        logger.exception(f"Невідома помилка при  завантажені моделі Spacy : {e}")
        raise RuntimeError(f"Невідома помилка при  завантажені моделі Spacy : {e}")
    yield
    logger.info("Додаток FastAPI завершує роботу. Вивільнення ресурсів (якщо потрібно).")

app = FastAPI(
        title="Weapon Entity Recognition API",
        description= "API для виділення сутностей з тексту ща допомогою NER, заснованої на spacy(3.8.5)",
        lifespan=lifespan
)

@app.get("/health", summary="Checking status of service")
async def health_check():
    if nlp is not None and isinstance(nlp, spacy.Language):
        return {
            "status": "ok",
            "model_status": "loaded",
            "message": "SpaCy model loaded successfully and application is operational.",
            "app_version": "1.0.0", 
            "model_name": nlp.meta.get("name", "unknown_model"), 
            "model_version": nlp.meta.get("version", "unknown_version"), 
            "timestamp": datetime.utcnow().isoformat() + "Z" 
        }
    else:
        logger.error("Health check: SpaCy model is not loaded. Application is unhealthy.")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "model_status": "not_loaded",
                "message": "SpaCy model failed to load during startup or is not ready.",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
           


@app.post("/analyze_fabula", response_model=FabulaOutput)
async def analyze_fabula(input_data: FabulaInput):
    fabula = input_data.fabula
    results = FabulaOutput()
    try:
        doc = nlp(fabula)
        results.entities = [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        logger.error(f"Error working with text in fab '{fabula}'. Exception:{e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Помилка обробки тексту: {e}")
    return results