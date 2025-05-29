from fastapi import FastAPI, HTTPException, Request, status
from typing import List, Tuple
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi.responses import JSONResponse, StreamingResponse 
import time 
from spacy.matcher import Matcher
import spacy
import json as js 
import logging
import os
import matplotlib.pyplot as plt
import io

from pyd_models import FabulaInput, FabulaOutput, FabulaBatchOutputItem, FabulaBatchInput, FabulaBatchOutput, WeaponSerialNumber
from stats import StatsManager
# --- Налаштування логування ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Конфігурація ---
MODEL_PATH  = os.getenv("MODEL_PATH", ".app/model-best") 
STATS_FILE = "api_processing_stats.json" 


NUMERIC_CALIBER_PART_REGEX = r"(?:\d{1,2}(?:[.,]\d{1,3})?|\.\d{2,3}|(?:4|8|10|12|16|20|24|28|32|36))"
SECOND_NUM_PART_REGEX = r"\d{2,3}[A-Za-zА-Яа-я]?"

SERIAL_TOKEN_REGEX = r"^[A-Za-zА-Яа-яЇїІіЄєҐґ0-9](?:[A-Za-zА-Яа-яЇїІіЄєҐґ0-9./-]{0,23}[A-Za-zА-Яа-яЇїІіЄєҐґ0-9])?$|^[A-Za-zА-Яа-яЇїІіЄєҐґ0-9]{2,25}$"
SERIAL_PREFIX_REGEX = r"^[A-Za-zА-Яа-яЇїІіЄєҐґ]{2,7}$"
SERIAL_SUFFIX_REGEX = r"^(?=[A-Za-zА-Яа-яЇїІіЄєҐґ0-9./-]*[0-9])[A-Za-zА-Яа-яЇїІіЄєҐґ0-9./-]{2,20}$|^[0-9./-]{2,20}$"


sn_token_single_def = {"TEXT": {"REGEX": SERIAL_TOKEN_REGEX}}
sn_token_prefix_def = {"TEXT": {"REGEX": SERIAL_PREFIX_REGEX}}
sn_token_suffix_def = {"TEXT": {"REGEX": SERIAL_SUFFIX_REGEX}}

markers_text_lower = ["номер", "ном", "н", "маркування"]
markers_symbols = ["№", "#"]
weapon_number_patterns_config = []

for marker_sym in markers_symbols:
    weapon_number_patterns_config.append([{"TEXT": marker_sym}, {"TEXT": ":", "OP": "?"}, sn_token_single_def])
    weapon_number_patterns_config.append([{"TEXT": marker_sym}, sn_token_single_def])
    weapon_number_patterns_config.append([{"TEXT": marker_sym}, {"TEXT": ":", "OP": "?"}, sn_token_prefix_def, sn_token_suffix_def])
    weapon_number_patterns_config.append([{"TEXT": marker_sym}, sn_token_prefix_def, sn_token_suffix_def])

for marker_txt in markers_text_lower:
    weapon_number_patterns_config.append([{"LOWER": marker_txt}, {"TEXT": {"IN": [":", "."]}, "OP": "?"}, sn_token_single_def])
    weapon_number_patterns_config.append([{"LOWER": marker_txt}, sn_token_single_def])
    weapon_number_patterns_config.append([{"LOWER": marker_txt}, {"TEXT": {"IN": [":", "."]}, "OP": "?"}, sn_token_prefix_def, sn_token_suffix_def])
    weapon_number_patterns_config.append([{"LOWER": marker_txt}, sn_token_prefix_def, sn_token_suffix_def])
    if marker_txt in ["ном", "н"]:
        weapon_number_patterns_config.append([{"LOWER": marker_txt}, {"TEXT": "."}, sn_token_single_def])
        weapon_number_patterns_config.append([{"LOWER": marker_txt}, {"TEXT": "."}, sn_token_prefix_def, sn_token_suffix_def])
nlp: spacy.language.Language = None
matcher = None
stats_manager: StatsManager = None 


# --- Lifespan Event Handlers ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp, stats_manager, matcher
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
        matcher = Matcher(nlp.vocab)
        if not weapon_number_patterns_config:
            logger.warning("Патерни для пошуку номерів зброї не визначені! Matcher буде порожнім.")
        else:
            for i, pattern_group in enumerate(weapon_number_patterns_config):
                try:
                    matcher.add(f"WEAPON_NUMBER_PATTERN_{i}", [pattern_group])
                except Exception as e:
                    logger.error(f"Не вдалося додати патерн номеру зброї {i}: {pattern_group}. Помилка: {e}")
            logger.info(f"Matcher ініціалізовано з {len(matcher)} патернами номерів зброї.")
    except OSError:
        logger.exception(f"Не вдалося завантажити модель Spacy з {MODEL_PATH}. Переконайтеся, що модель існує за цим шляхом.")
        raise RuntimeError(f"Не вдалося завантажити модель Spacy з {MODEL_PATH}") 
    except Exception as e :
        logger.exception(f"Невідома помилка при завантажені моделі Spacy : {e}")
        raise RuntimeError(f"Невідома помилка при завантажені моделі Spacy : {e}")


    yield 

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
    global nlp, matcher
    if nlp is None:
        logger.error("'/analyze_fabula' called before model was loaded.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Модель SpaCy ще не завантажена або недоступна. Спробуйте пізніше.")
    if matcher is None: # Додаткова перевірка для matcher
        logger.error("'/analyze_fabula' викликано до ініціалізації Matcher.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Matcher для номерів зброї не ініціалізовано.")

    
    # Перевіряємо, чи менеджер статистики ініціалізований перед обробкою
    # Якщо він не критичний для роботи ендпоінта, можна просто залогувати і продовжити
    if stats_manager is None:
         logger.warning("StatsManager не ініціалізовано. Статистика не буде оновлена для цієї фабули.")
         # Можна продовжити без оновлення статистики
         pass # Продовжуємо виконувати обробку тексту моделлю

    fabula = input_data.fabula
    results = FabulaOutput()
    processing_duration = 0.0
    CONTEXT_WINDOW_SIZE = 25

    try:
        start_time = time.time()
        doc = nlp(fabula) 
        results.entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Список для зберігання ВСІХ знайдених відповідностей для ВСІХ сутностей,
        # ПІСЛЯ фільтрації на "найповніший" для кожного контекстного вікна
        all_final_serial_number_matches: List[WeaponSerialNumber] = []

        for ent in doc.ents: # Зовнішній цикл по сутностях
            window_start_token_idx = max(0, ent.start - CONTEXT_WINDOW_SIZE)
            window_end_token_idx = min(len(doc), ent.end + CONTEXT_WINDOW_SIZE)
            context_span = doc[window_start_token_idx:window_end_token_idx]
            
            matches_in_context_raw = matcher(context_span) # Отримуємо "сирі" збіги
            
            # Список для зберігання кандидатів на серійні номери для ПОТОЧНОГО context_span
            candidate_matches_for_current_span = []

            for match_id_hash, start_token_in_span, end_token_in_span in matches_in_context_raw:
                matched_segment = context_span[start_token_in_span:end_token_in_span]
                original_segment_start_char  = context_span[start_token_in_span].idx
                original_segment_end_char = context_span[end_token_in_span - 1].idx + len(context_span[end_token_in_span - 1].text)
                
                extracted_serial = ""
                try:
                    pattern_key_str = nlp.vocab.strings[match_id_hash]
                    pattern_index  = int(pattern_key_str.split("_")[-1])
                    matched_pattern_structure =  weapon_number_patterns_config[pattern_index]
                    num_serial_tokens = 0
                    for token_pattern_dict in reversed(matched_pattern_structure):
                        if "TEXT" in token_pattern_dict and isinstance(token_pattern_dict["TEXT"], dict) and "REGEX" in token_pattern_dict["TEXT"]:
                            num_serial_tokens +=1
                        elif not("TEXT" in token_pattern_dict and isinstance(token_pattern_dict["TEXT"], dict) and "REGEX" in token_pattern_dict["TEXT"]):
                            if num_serial_tokens > 0:
                                break
                    if num_serial_tokens > 0 and len(matched_segment) >= num_serial_tokens:
                        serial_tokens_span = matched_segment[-num_serial_tokens:]
                        extracted_serial = serial_tokens_span.text
                    else:
                        extracted_serial = matched_segment.text
                except Exception as e_extraction:
                    logger.warning(f"Помилка виділення серійного номера з '{matched_segment.text}': {e_extraction}")
                    extracted_serial = matched_segment.text
                
                candidate_matches_for_current_span.append({
                    'full_matched_text': matched_segment.text,
                    'original_segment_start_char': original_segment_start_char,
                    'original_segment_end_char': original_segment_end_char,
                    'extracted_serial': extracted_serial
                })

            # Якщо для поточного context_span не знайдено кандидатів, переходимо до наступної сутності
            if not candidate_matches_for_current_span:
                continue

            # Сортуємо кандидатів:
            # 1. За початковим символом (зростання)
            # 2. За довжиною збігу (спадання, тобто довші перші для однакового початку)
            candidate_matches_for_current_span.sort(
                key=lambda m: (m['original_segment_start_char'], -(m['original_segment_end_char'] - m['original_segment_start_char']))
            )

            # Фільтруємо, щоб залишити "найповніші"
            selected_matches_for_this_span = []
            if candidate_matches_for_current_span:
                # Завжди додаємо перший кандидат (він найдовший для своєї початкової позиції)
                selected_matches_for_this_span.append(candidate_matches_for_current_span[0])

                for current_candidate_data in candidate_matches_for_current_span[1:]:
                    last_added_match_data = selected_matches_for_this_span[-1]

                    # Сценарій 1: Поточний кандидат починається там же, де й останній доданий.
                    # Оскільки сортували за довжиною (спадання), поточний буде коротшим або таким самим. Ігноруємо.
                    if current_candidate_data['original_segment_start_char'] == last_added_match_data['original_segment_start_char']:
                        continue

                    # Сценарій 2: Поточний кандидат починається ПІСЛЯ того, як останній доданий закінчився (немає перекриття).
                    # Це новий, окремий потенційний номер. Додаємо його.
                    if current_candidate_data['original_segment_start_char'] >= last_added_match_data['original_segment_end_char']:
                        selected_matches_for_this_span.append(current_candidate_data)
                        continue
                    
                    # Сценарій 3: Поточний кандидат перекривається з останнім доданим, але починається пізніше.
                    # (current_candidate_data['original_segment_start_char'] < last_added_match_data['original_segment_end_char'] та
                    #  current_candidate_data['original_segment_start_char'] > last_added_match_data['original_segment_start_char'])
                    # Якщо поточний кандидат повністю міститься в останньому доданому (тобто закінчується не пізніше), ігноруємо.
                    if current_candidate_data['original_segment_end_char'] <= last_added_match_data['original_segment_end_char']:
                        continue
            for final_match_data in selected_matches_for_this_span:
                serial_number_match = WeaponSerialNumber(
                    entity_text= ent.text,
                    entity_label= ent.label_,
                    entity_start_ch= ent.start_char,
                    entity_end_ch= ent.end_char,
                    matched_weapon_segment_text=final_match_data['full_matched_text'],
                    matched_segment_start_char= final_match_data['original_segment_start_char'],
                    matched_segment_end_char=final_match_data['original_segment_end_char'],
                    serial_number=final_match_data['extracted_serial']
                )
                all_final_serial_number_matches.append(serial_number_match)
        
        # Присвоєння фінального списку результатам
        results.weapon_serial_numbers = all_final_serial_number_matches
        end_time = time.time()
        processing_duration = end_time - start_time

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

