from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any
import spacy
import json as js 
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, date 
from fastapi.responses import JSONResponse, StreamingResponse # Імпортуємо StreamingResponse для повернення зображення
import time 
from pathlib import Path 

# --- Імпорти для графіків ---
import matplotlib.pyplot as plt
import io # Для збереження графіка у буфер пам'яті
# Необхідно переконатися, що Matplotlib встановлено: pip install matplotlib


# --- Налаштування логування ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Конфігурація ---
MODEL_PATH  = os.getenv("MODEL_PATH", ".app/model-best") 
STATS_FILE = "api_processing_stats.json" 

# --- Клас для управління статистикою (збереження/завантаження з файлу) ---
class StatsManager:
    def __init__(self, stats_file: str = STATS_FILE):
        self._stats_file = Path(stats_file)
        
        # Ініціалізуємо статистику, намагаючись завантажити з файлу
        loaded_stats = self._load_from_file()

        # Загальна статистика
        self._total_processed = loaded_stats.get("total_fabulas_processed", 0)
        self._total_entities_found = loaded_stats.get("total_entities_found", 0)
        self._entities_by_label = loaded_stats.get("entities_by_label", {})
        if not isinstance(self._entities_by_label, dict): self._entities_by_label = {}

        self._total_processing_time_seconds = loaded_stats.get("total_processing_time_seconds", 0.0)

        # Щоденна статистика
        last_update_date_str = loaded_stats.get("daily_stats_date", None)
        last_update_date = None
        if last_update_date_str:
            try:
                last_update_date = date.fromisoformat(last_update_date_str)
            except ValueError:
                logger.warning(f"Некоректний формат дати '{last_update_date_str}' у файлі статистики. Використовуємо сьогоднішню дату для щоденної статистики.")
                last_update_date = date.today()
        else:
             last_update_date = date.today()


        current_date = date.today()

        if current_date > last_update_date:
            logger.info(f"Настав новий день ({current_date}). Скидаємо щоденну статистику.")
            self._daily_processed = 0
            self._daily_entities_found = 0
            self._daily_entities_by_label = {}
            self._daily_processing_time_seconds = 0.0
            self._last_update_date = current_date 

        else:
            daily_stats_loaded = loaded_stats.get("daily_stats", {})
            self._daily_processed = daily_stats_loaded.get("processed_fabulas", 0)
            self._daily_entities_found = daily_stats_loaded.get("found_weapon_entities", 0)
            self._daily_entities_by_label = daily_stats_loaded.get("entities_by_label", {})
            if not isinstance(self._daily_entities_by_label, dict): self._daily_entities_by_label = {}
            self._daily_processing_time_seconds = daily_stats_loaded.get("total_processing_time_seconds", 0.0)
            self._last_update_date = last_update_date 


        logger.info("StatsManager ініціалізовано (включно із завантаженням).")


    def _load_from_file(self) -> Dict[str, Any]:
        """Намагається завантажити статистику з JSON файлу. Повертає словник."""
        default_stats_structure = {
             "total_fabulas_processed": 0,
             "total_entities_found": 0,
             "entities_by_label": {},
             "total_processing_time_seconds": 0.0,
             "average_processing_time_per_fabula_ms": 0.0,
             "last_update_timestamp": None, 
             "daily_stats": { 
                 "date": str(date.today()), 
                 "processed_fabulas": 0,
                 "found_weapon_entities": 0,
                 "entities_by_label": {},
                 "total_processing_time_seconds": 0.0,
                 "average_processing_time_per_fabula_ms": 0.0,
             },
            "daily_stats_date": str(date.today()),
        }

        stats = default_stats_structure.copy()

        if self._stats_file.exists():
            try:
                with open(self._stats_file, 'r', encoding='utf-8') as f:
                    loaded_stats = js.load(f)
                    for key, default_value in default_stats_structure.items():
                         if isinstance(default_value, dict) and key in loaded_stats and isinstance(loaded_stats[key], dict):
                              for nested_key, nested_default_value in default_value.items():
                                   stats[key][nested_key] = loaded_stats[key].get(nested_key, nested_default_value)
                         else:
                            stats[key] = loaded_stats.get(key, default_value)

                logger.info(f"Статистику успішно завантажено з: {self._stats_file}")
            except (js.JSONDecodeError, Exception) as e:
                logger.warning(f"Не вдалося завантажити статистику з {self._stats_file}. Початок з нуля. Помилка: {e}")
        else:
            logger.info(f"Файл статистики {self._stats_file} не знайдено. Початок з нуля.")

        if not isinstance(stats.get("entities_by_label"), dict): stats["entities_by_label"] = {}
        if not isinstance(stats.get("daily_stats", {}).get("entities_by_label"), dict): stats.get("daily_stats", {})["entities_by_label"] = {}


        return stats


    def save_to_file(self):
        """Зберігає поточну статистику у JSON файл."""
        try:
            avg_total_time_ms = (self._total_processing_time_seconds / self._total_processed) * 1000 if self._total_processed > 0 else 0
            avg_daily_time_ms = (self._daily_processing_time_seconds / self._daily_processed) * 1000 if self._daily_processed > 0 else 0

            stats_to_save = {
                "total_fabulas_processed": self._total_processed,
                "total_entities_found": self._total_entities_found,
                "entities_by_label": dict(self._entities_by_label),
                "total_processing_time_seconds": round(self._total_processing_time_seconds, 2),
                "average_processing_time_per_fabula_ms": round(avg_total_time_ms, 2),
                "last_update_timestamp": datetime.utcnow().isoformat() + "Z",
                "daily_stats": { 
                    "date": str(self._last_update_date), 
                    "processed_fabulas": self._daily_processed,
                    "found_weapon_entities": self._daily_entities_found,
                    "entities_by_label": dict(self._daily_entities_by_label),
                    "total_processing_time_seconds": round(self._daily_processing_time_seconds, 2),
                    "average_processing_time_per_fabula_ms": round(avg_daily_time_ms, 2),
                },
                "daily_stats_date": str(self._last_update_date), 
            }

            self._stats_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self._stats_file, 'w', encoding='utf-8') as f:
                js.dump(stats_to_save, f, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.error(f"Помилка при збереженні статистики до {self._stats_file}: {e}")


    def _check_and_reset_daily(self):
        """
        Перевіряє дату і скидає щоденну статистику, якщо настав новий день.
        Викликає збереження файлу при скиданні.
        """
        current_date = date.today()
        if current_date > self._last_update_date:
            logger.info(f"Настав новий день ({current_date}). Скидаємо щоденну статистику.")

            self._daily_processed = 0
            self._daily_entities_found = 0
            self._daily_entities_by_label = {}
            self._daily_processing_time_seconds = 0.0
            self._last_update_date = current_date 

            # Зберігаємо файл одразу після скидання
            self.save_to_file() 


    def update_stats(self, doc: spacy.tokens.Doc, processing_time_seconds: float):
        """
        Оновлює статистику після обробки одного документа. 
        Викликає перевірку дати на початку та збереження файлу в кінці.
        """
        self._check_and_reset_daily() 

        self._total_processed += 1
        self._total_processing_time_seconds += processing_time_seconds

        self._daily_processed += 1
        self._daily_processing_time_seconds += processing_time_seconds;

        found_entities_count = 0
        entities_in_doc_by_label = {} 

        for ent in doc.ents:
            if ent.label_ and ent.label_.startswith("WEAPON"):
                found_entities_count += 1
                entities_in_doc_by_label[ent.label_] = entities_in_doc_by_label.get(ent.label_, 0) + 1


        self._total_entities_found += found_entities_count
        self._daily_entities_found += found_entities_count

        for label, count in entities_in_doc_by_label.items():
            self._entities_by_label[label] = self._entities_by_label.get(label, 0) + count
            self._daily_entities_by_label[label] = self._daily_entities_by_label.get(label, 0) + count

        # Зберігаємо файл після кожного оновлення
        self.save_to_file()


    def get_stats(self) -> Dict[str, Any]:
        """Повертає поточну статистику."""
        self._check_and_reset_daily() 

        avg_total_time_ms = (self._total_processing_time_seconds / self._total_processed) * 1000 if self._total_processed > 0 else 0
        avg_daily_time_ms = (self._daily_processing_time_seconds / self._daily_processed) * 1000 if self._daily_processed > 0 else 0

        stats_payload = {
            "total_stats": {
                "processed_fabulas": self._total_processed,
                "found_weapon_entities": self._total_entities_found,
                "entities_by_label": dict(self._entities_by_label), 
                "total_processing_time_seconds": round(self._total_processing_time_seconds, 2),
                "average_processing_time_per_fabula_ms": round(avg_total_time_ms, 2),
            },
            "daily_stats": {
                "date": str(self._last_update_date), 
                "processed_fabulas": self._daily_processed,
                "found_weapon_entities": self._daily_entities_found,
                "entities_by_label": dict(self._daily_entities_by_label), 
                "total_processing_time_seconds": round(self._daily_processing_time_seconds, 2),
                "average_processing_time_per_fabula_ms": round(avg_daily_time_ms, 2),
            }
        }

        return stats_payload


# --- Глобальні змінні ---
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
         # Якщо ініціалізація менеджера статистики критична для роботи, можна тут зупинити додаток:
         # raise RuntimeError(f"Не вдалося ініціалізувати StatsManager: {e}")
         # Інакше, додаток продовжить роботу без статистики.


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

# --- Моделі запитів та відповідей Pydantic ---
class FabulaInput(BaseModel):
    fabula: str

class FabulaOutput(BaseModel):
    entities: List[Tuple[str, str]] = []

# --- API Endpoints ---

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

# --- Endpoint для отримання статистики (JSON) ---
@app.get("/stat", summary="Get processing statistics (JSON)")
async def get_processing_statistics_json():
    if stats_manager is None:
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
