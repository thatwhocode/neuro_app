from fastapi import FastAPI, HTTPException, Request, status
from typing import List, Tuple
import spacy
import json as js 
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, date 
from fastapi.responses import JSONResponse, StreamingResponse 
import time 
from pathlib import Path 
# --- Імпорти для графіків ---
import matplotlib.pyplot as plt
import io

from pyd_models import FabulaInput, FabulaOutput, FabulaBatchOutputItem, FabulaBatchInput, FabulaBatchOutput
from stats import StatsManager
# --- Налаштування логування ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Конфігурація ---
MODEL_PATH  = os.getenv("MODEL_PATH", ".app/model-best") 
STATS_FILE = "api_processing_stats.json" 



nlp: spacy.language.Language = None
stats_manager: StatsManager = None 


# --- Lifespan Event Handlers ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp, stats_manager
    logger.info("Додаток FastAPI запускається...")

    # 1. Ініціалізація StatsManager (він сам завантажить статистику з файлу)
    try:
        stats_manager = StatsManager(stats_file=STATS_FILE)
        logger.info("StatsManager успішно ініціалізовано.")
    except Exception as e:
         logger.error(f"Критична помилка при ініціалізації StatsManager: {e}", exc_info=True)



    # 2. Завантаження моделі SpaCy
    try:
        logger.info(f"Спроба завантажити модель Spacy з : {MODEL_PATH} ")
        nlp = spacy.load(MODEL_PATH)
        logger.info("Модель SpaCy успішно завантажена.")
    except OSError:
        logger.exception(f"Не вдалося завантажити модель Spacy з {MODEL_PATH}. Переконайтеся, що модель існує за цим шляхом.")
        raise RuntimeError(f"Не вдалося завантажити модель Spacy з {MODEL_PATH}") 
    except Exception as e :
        logger.exception(f"Невідома помилка при завантажені моделі Spacy : {e}")
        raise RuntimeError(f"Невідома помилка при завантажені моделі Spacy : {e}")


    yield # Додаток готовий приймати запити

    # --- Логіка завершення роботи (shutdown) ---
    logger.info("Додаток FastAPI завершує роботу. Вивільнення ресурсів...")
    # 3. Збереження статистики перед зупинкою
    if stats_manager:
        try:
            stats_manager.save_to_file() 
            logger.info("Статистика збережена під час завершення роботи.")
        except Exception as e:
            logger.error(f"Помилка при збереженні статистики під час завершення роботи: {e}")
    
    logger.info("Ресурси вивільнені. Додаток зупинено.")


# --- Екземпляр FastAPI додатку ---
app = FastAPI(
        title="Weapon Entity Recognition API",
        description= "API для виділення сутностей з тексту за допомогою NER, заснованої на spacy(3.8.5)",
        lifespan=lifespan 
)

#--- API Endpoints ---

@app.get("/health", summary="Checking status of service")
async def health_check():
    model_loaded = nlp is not None and isinstance(nlp, spacy.Language)
    stats_manager_initialized = stats_manager is not None

    if model_loaded and stats_manager_initialized:
        return {
            "status": "ok",
            "model_status": "loaded",
            "stats_manager_status": "initialized",
            "message": "SpaCy model loaded and StatsManager initialized. Application is operational.",
            "app_version": "1.0.0", 
            "model_name": nlp.meta.get("name", "unknown_model") if model_loaded else "N/A",
            "model_version": nlp.meta.get("version", "unknown_version") if model_loaded else "N/A",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    else:
        logger.error("Health check failed: SpaCy model or StatsManager not ready.")
        status_details = []
        if not model_loaded:
            status_details.append("SpaCy model not loaded")
        if not stats_manager_initialized:
            status_details.append("StatsManager not initialized")

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            content={
                "status": "error",
                "message": "Application is unhealthy. Required components not ready.",
                "details": status_details,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

@app.post("/analyze_fabula", response_model=FabulaOutput, summary="Analyze fabula text for weapon entities")
async def analyze_fabula(input_data: FabulaInput):
    if nlp is None:
        logger.error("'/analyze_fabula' called before model was loaded.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Модель SpaCy ще не завантажена або недоступна. Спробуйте пізніше.")
    
    # Перевіряємо, чи менеджер статистики ініціалізований перед обробкою
    # Якщо він не критичний для роботи ендпоінта, можна просто залогувати і продовжити
    if stats_manager is None:
         logger.warning("StatsManager не ініціалізовано. Статистика не буде оновлена для цієї фабули.")
         # Можна продовжити без оновлення статистики
         pass # Продовжуємо виконувати обробку тексту моделлю

    fabula = input_data.fabula
    results = FabulaOutput()
    processing_duration = 0.0

    try:
        start_time = time.time()
        doc = nlp(fabula) 
        end_time = time.time()
        processing_duration = end_time - start_time

        results.entities = [(ent.text, ent.label_) for ent in doc.ents]

        # --- Оновлюємо статистику (тільки якщо stats_manager ініціалізовано) ---
        if stats_manager: 
            stats_manager.update_stats(doc, processing_duration)
            # Збереження файлу відбувається всередині update_stats

    except Exception as e:
        logger.error(f"Помилка обробки тексту в '/analyze_fabula': '{fabula[:100]}...'. Виняток: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Помилка обробки тексту: {e}")

    
    return results
@app.post("/analyze_fabulas_batch", response_model=FabulaBatchOutput, summary="Analyze a batch of fabulas")
async def analyze_fabulas_batch(batch_input: FabulaBatchInput):
    if nlp is None: 
        logger .error("'/analyze_fabulas_batch' was called before model was loaded")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Модель Spacy не завантажено, спробуйте пізніше") 
    else:
        logger.info("Model succesfully initialized")
    if StatsManager is None: 
        logger.warning("StatsManager не ініціалізовано. Статистика не буде оновлена")
        pass
    else:
        logger.info("StatsManager succesfully initialized")
        
        
    batch_results: List[FabulaBatchOutputItem] = []
    processed_ctr: int = 0
    failed_ctr: int  = 0
    for fabula_input_item in batch_input.fabulas:
        fabula = fabula_input_item.fabula
        processing_duration  = 0.0
        entities_for_item: List[Tuple[str, str]] =[]
        item_failed = False
        try:
            start_time = time.time()
            doc = nlp(fabula)
            end_time = time.time()
            processing_duration = end_time - start_time
            if stats_manager:
                stats_manager.update_stats(doc, processing_duration)
            batch_results.append(FabulaBatchOutputItem(
                entities= entities_for_item
                
            ))
            processed_ctr+=1
        except Exception as e: 
            item_failed = True
            failed_ctr+= 1 
            logger.error(f"Помилка обробки тексту{e}")
    logger.info(f"Завершено обробку батчу: успішно{processed_ctr}, помилок{failed_ctr}")
    return FabulaBatchOutput(results=batch_results, processed_ctr=processed_ctr, failed_ctr=item_failed)




# --- Endpoint для отримання статистики (JSON) ---
@app.get("/stat", summary="Get processing statistics (JSON)")
async def get_processing_statistics_json():
    if StatsManager is None:
        logger.error("'/stat' called before StatsManager was initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Система статистики ще не готова або недоступна. Спробуйте пізніше.")
    try:
        current_stats = stats_manager.get_stats()
        return JSONResponse(content=current_stats)
    except Exception as e:
        logger.error(f"Помилка при отриманні статистики в '/stat': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Помилка при отриманні статистики: {e}")
    


# --- Новий endpoint для отримання графіка статистики (PNG) ---
@app.get("/stats/entities_by_label_plot", summary="Get a plot of total weapon entities found by label (PNG)")
async def get_entities_by_label_plot():
    if stats_manager is None:
        logger.error("'/stats/entities_by_label_plot' called before StatsManager was initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Система статистики ще не готова або недоступна. Спробуйте пізніше.")

    try:
        # Отримуємо поточну статистику з менеджера
        current_stats = stats_manager.get_stats()
        entities_by_label = current_stats['total_stats'].get('entities_by_label', {}) # Використовуємо .get() для безпеки

        if not entities_by_label:
            # Обробляємо випадок, коли статистика порожня
            return JSONResponse(content={"message": "Статистика сутностей за мітками ще не доступна або порожня."},
                                status_code=status.HTTP_404_NOT_FOUND)

        # Сортуємо сутності за кількістю для кращої візуалізації
        sorted_entities = dict(sorted(entities_by_label.items(), key=lambda item: item[1], reverse=True))

        labels = list(sorted_entities.keys())
        counts = list(sorted_entities.values())

        # Створюємо графік
        plt.figure(figsize=(10, 6)) # Розмір фігури
        plt.bar(labels, counts, color='skyblue')
        plt.xlabel("Мітка сутності")
        plt.ylabel("Кількість знайдених сутностей")
        plt.title("Загальна кількість знайденої зброї за мітками")
        plt.xticks(rotation=45, ha='right') # Обертаємо підписи по осі X, якщо вони довгі
        plt.tight_layout() # Автоматичне налаштування розмірів елементів, щоб вони не перекривалися

        # Зберігаємо графік у буфер пам'яті у форматі PNG
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0) # Переміщуємо курсор на початок буфера
        plt.close() # Закриваємо фігуру графіка, щоб звільнити пам'ять

        # Повертаємо вміст буфера як StreamingResponse з типом зображення PNG
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Помилка при генерації графіка в '/stats/entities_by_label_plot': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Помилка при генерації графіка: {e}")

