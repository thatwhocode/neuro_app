from fastapi import FastAPI, HTTPException, Request, status
from typing import List, Tuple, Optional, Any 
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi.responses import JSONResponse, StreamingResponse 
import time 
from spacy.matcher import Matcher
import spacy
import re # Переконайтеся, що re імпортовано

import logging
import os
import matplotlib.pyplot as plt
import io

from pyd_models import (
    FabulaInput, FabulaOutput, 
    FabulaBatchOutputItem, FabulaBatchInput, FabulaBatchOutput, 
    WeaponSerialNumber, WeaponCaliberMatch 
)
from app_config import(
    MODEL_PATH, STATS_FILE, CONTEXT_WINDOW_SIZE, 
    SERIAL_TOKEN_REGEX, SERIAL_PREFIX_REGEX, SERIAL_SUFFIX_REGEX,
    NUMERIC_CALIBER_PART_REGEX, SECOND_NUM_PART_REGEX
    )
from app_config import(sn_token_single_def, sn_token_prefix_def, sn_token_suffix_def)
from app_paterns import weapon_number_patterns_config, caliber_patterns_config
from stats import StatsManager 

# --- Налаштування логування ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Глобальні змінні ---
nlp: spacy.language.Language = None
serial_number_matcher: Matcher = None 
caliber_matcher: Matcher = None      
stats_manager: StatsManager = None 

# --- Lifespan Event Handlers ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp, stats_manager, serial_number_matcher, caliber_matcher
    logger.info("Додаток FastAPI запускається...")

    try:
        stats_manager = StatsManager(stats_file=STATS_FILE)
        logger.info("StatsManager успішно ініціалізовано.")
    except Exception as e:
         logger.error(f"Критична помилка при ініціалізації StatsManager: {e}", exc_info=True)

    try:
        logger.info(f"Спроба завантажити модель Spacy з : {MODEL_PATH} ")
        nlp = spacy.load(MODEL_PATH)
        logger.info("Модель SpaCy успішно завантажена.")
        
        serial_number_matcher = Matcher(nlp.vocab)
        if not weapon_number_patterns_config:
            logger.warning("Патерни для пошуку номерів зброї не визначені! Matcher для номерів буде порожнім.")
        else:
            for i, pattern_group_sn in enumerate(weapon_number_patterns_config):
                try:
                    serial_number_matcher.add(f"WEAPON_NUMBER_PATTERN_{i}", [pattern_group_sn])
                except Exception as e:
                    logger.error(f"Не вдалося додати патерн номеру зброї {i}: {pattern_group_sn}. Помилка: {e}", exc_info=True)
            logger.info(f"Matcher для серійних номерів ініціалізовано з {len(serial_number_matcher)} патернами.")

        caliber_matcher = Matcher(nlp.vocab)
        if not caliber_patterns_config:
            logger.warning("Патерни для пошуку калібрів не визначені! Matcher для калібрів буде порожнім.")
        else:
            for i, pattern_group_with_flags in enumerate(caliber_patterns_config):
                cleaned_pattern_group_for_matcher = []
                for token_def_with_flags in pattern_group_with_flags:
                    cleaned_token_def = {
                        k: v for k, v in token_def_with_flags.items() 
                        if not k.startswith('_') 
                    }
                    cleaned_pattern_group_for_matcher.append(cleaned_token_def)
                try:
                    caliber_matcher.add(f"CALIBER_PATTERN_{i}", [cleaned_pattern_group_for_matcher])
                except Exception as e:
                    logger.error(f"Не вдалося додати очищений патерн калібру {i} (початковий: {pattern_group_with_flags}). Помилка: {e}", exc_info=True)
            logger.info(f"Matcher для калібрів ініціалізовано з {len(caliber_matcher)} патернами.")

    except OSError:
        logger.exception(f"Не вдалося завантажити модель Spacy з {MODEL_PATH}.")
        raise RuntimeError(f"Не вдалося завантажити модель Spacy з {MODEL_PATH}") 
    except Exception as e :
        logger.exception(f"Невідома помилка при завантажені моделі Spacy або ініціалізації Matcher'ів: {e}")
        raise RuntimeError(f"Невідома помилка при завантажені моделі Spacy або ініціалізації Matcher'ів: {e}")

    yield 

    logger.info("Додаток FastAPI завершує роботу. Вивільнення ресурсів...")
    if stats_manager:
        try:
            stats_manager.save_to_file() 
            logger.info("Статистика збережена під час завершення роботи.")
        except Exception as e:
            logger.error(f"Помилка при збереженні статистики під час завершення роботи: {e}")
    logger.info("Ресурси вивільнені. Додаток зупинено.")

app = FastAPI(
        title="Weapon Entity Recognition API",
        description= "API для виділення сутностей з тексту за допомогою NER та атрибутів зброї.",
        lifespan=lifespan 
)

#--- API Endpoints ---
@app.get("/health", summary="Checking status of service")
async def health_check():
    model_loaded = nlp is not None
    s_matcher_loaded = serial_number_matcher is not None
    c_matcher_loaded = caliber_matcher is not None
    stats_manager_initialized = stats_manager is not None

    if model_loaded and stats_manager_initialized and s_matcher_loaded and c_matcher_loaded:
        return {
            "status": "ok",
            "message": "Application is operational.",
            "components": {
                "model_status": "loaded",
                "serial_number_matcher_status": "loaded",
                "caliber_matcher_status": "loaded",
                "stats_manager_status": "initialized",
            },
            "app_version": "1.0.2", # Оновлено версію
            "model_name": nlp.meta.get("name", "unknown_model"),
            "model_version": nlp.meta.get("version", "unknown_version"),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    else:
        logger.error("Health check failed: SpaCy model, Matcher(s) or StatsManager not ready.")
        status_details = []
        if not model_loaded: status_details.append("SpaCy model not loaded")
        if not s_matcher_loaded: status_details.append("Serial number matcher not loaded")
        if not c_matcher_loaded: status_details.append("Caliber matcher not loaded")
        if not stats_manager_initialized: status_details.append("StatsManager not initialized")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": "Application is unhealthy. Required components not ready.",
                "details": status_details,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

@app.post("/analyze_fabula", response_model=FabulaOutput, summary="Analyze fabula text for weapon entities, serial numbers, and calibers")
async def analyze_fabula(input_data: FabulaInput):
    global nlp, serial_number_matcher, caliber_matcher 

    if nlp is None:
        logger.error("'/analyze_fabula' called before model was loaded.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Модель SpaCy ще не завантажена.")
    if serial_number_matcher is None:
        logger.error("'/analyze_fabula' викликано до ініціалізації Matcher для серійних номерів.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Matcher для серійних номерів не ініціалізовано.")
    if caliber_matcher is None:
        logger.error("'/analyze_fabula' викликано до ініціалізації Matcher для калібрів.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Matcher для калібрів не ініціалізовано.")
    
    if stats_manager is None:
         logger.warning("StatsManager не ініціалізовано. Статистика не буде оновлена для цієї фабули.")
         pass 

    fabula = input_data.fabula
    results = FabulaOutput()


    try:
        start_time = time.time()
        doc = nlp(fabula) 
        results.entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # --- Обробка СЕРІЙНИХ НОМЕРІВ ---
        all_final_serial_number_matches: List[WeaponSerialNumber] = []
        for ent in doc.ents:
            window_start_token_idx = max(0, ent.start - CONTEXT_WINDOW_SIZE)
            window_end_token_idx = min(len(doc), ent.end + CONTEXT_WINDOW_SIZE)
            context_span = doc[window_start_token_idx:window_end_token_idx]
            matches_in_context_raw_sn = serial_number_matcher(context_span)
            candidate_matches_for_current_span_sn = []
            for match_id_hash, start_token_in_span, end_token_in_span in matches_in_context_raw_sn:
                matched_segment = context_span[start_token_in_span:end_token_in_span]
                original_segment_start_char  = context_span[start_token_in_span].idx
                original_segment_end_char = context_span[end_token_in_span - 1].idx + len(context_span[end_token_in_span - 1].text)
                extracted_serial = ""
                try:
                    pattern_key_str = nlp.vocab.strings[match_id_hash]
                    pattern_index  = int(pattern_key_str.split("_")[-1]) 
                    matched_pattern_structure =  weapon_number_patterns_config[pattern_index]
                    num_serial_tokens = 0
                    # Евристика для виділення чистого номера: рахуємо токени з REGEX з кінця патерна
                    for token_pattern_dict in reversed(matched_pattern_structure):
                        # Перевіряємо, чи REGEX цього токена - один з тих, що визначають частини серійного номера
                        # Це припускає, що sn_token_single_def і т.д. використовуються безпосередньо в weapon_number_patterns_config
                        # і їхні словники містять саме ці рядки REGEX.
                        if  token_pattern_dict == sn_token_single_def or \
                            token_pattern_dict == sn_token_prefix_def or \
                            token_pattern_dict == sn_token_suffix_def:
                            num_serial_tokens +=1
                        # Якщо це не частина номера (наприклад, маркер або роздільник), зупиняємось
                        elif not (token_pattern_dict.get("TEXT", {}).get("REGEX") in [SERIAL_TOKEN_REGEX, SERIAL_PREFIX_REGEX, SERIAL_SUFFIX_REGEX]):
                            if num_serial_tokens > 0: break 
                    
                    if num_serial_tokens > 0 and len(matched_segment) >= num_serial_tokens:
                        serial_tokens_span = matched_segment[-num_serial_tokens:]
                        extracted_serial = serial_tokens_span.text
                    else: 
                        extracted_serial = matched_segment.text
                except Exception as e_extraction:
                    logger.warning(f"Помилка виділення серійного номера з '{matched_segment.text}': {e_extraction}", exc_info=True)
                    extracted_serial = matched_segment.text
                candidate_matches_for_current_span_sn.append({
                    'full_matched_text': matched_segment.text,
                    'original_segment_start_char': original_segment_start_char,
                    'original_segment_end_char': original_segment_end_char,
                    'extracted_serial': extracted_serial})
            if candidate_matches_for_current_span_sn:
                candidate_matches_for_current_span_sn.sort(
                    key=lambda m: (m['original_segment_start_char'], -(m['original_segment_end_char'] - m['original_segment_start_char'])))
                selected_matches_for_this_span_sn = [candidate_matches_for_current_span_sn[0]]
                for current_candidate_data in candidate_matches_for_current_span_sn[1:]:
                    last_added_match_data = selected_matches_for_this_span_sn[-1]
                    if current_candidate_data['original_segment_start_char'] == last_added_match_data['original_segment_start_char']: continue
                    if current_candidate_data['original_segment_start_char'] >= last_added_match_data['original_segment_end_char']:
                        selected_matches_for_this_span_sn.append(current_candidate_data)
                        continue
                    if current_candidate_data['original_segment_end_char'] <= last_added_match_data['original_segment_end_char']: continue
                    # Якщо перекривається, але не повністю міститься і не починається там же - можемо додати, якщо логіка це дозволяє
                    # Поточна логіка не додасть такий випадок, що спрощує до "найдовший на початку, потім непересічні"
                for final_match_data in selected_matches_for_this_span_sn:
                    all_final_serial_number_matches.append(WeaponSerialNumber(
                        entity_text=ent.text, entity_label=ent.label_, entity_start_ch=ent.start_char, entity_end_ch=ent.end_char,
                        matched_weapon_segment_text=final_match_data['full_matched_text'],
                        matched_segment_start_char=final_match_data['original_segment_start_char'],
                        matched_segment_end_char=final_match_data['original_segment_end_char'],
                        serial_number=final_match_data['extracted_serial']))
        results.weapon_serial_numbers = all_final_serial_number_matches
        
        # --- Обробка КАЛІБРІВ (оновлений блок) ---
        all_final_caliber_matches: List[WeaponCaliberMatch] = []
        for ent in doc.ents: 
            window_start_token_idx = max(0, ent.start - CONTEXT_WINDOW_SIZE)
            window_end_token_idx = min(len(doc), ent.end + CONTEXT_WINDOW_SIZE)
            context_span = doc[window_start_token_idx:window_end_token_idx]
            
            matches_in_context_raw_cal = caliber_matcher(context_span)
            candidate_matches_for_current_span_cal = []

            for match_id_hash, start_token_in_span, end_token_in_span in matches_in_context_raw_cal:
                matched_segment = context_span[start_token_in_span:end_token_in_span]
                original_segment_start_char = context_span[start_token_in_span].idx
                original_segment_end_char = context_span[end_token_in_span - 1].idx + len(context_span[end_token_in_span - 1].text)
                
                extracted_caliber = ""
                try:
                    pattern_key_str = nlp.vocab.strings[match_id_hash] 
                    pattern_index = int(pattern_key_str.split("_")[-1]) # CALIBER_PATTERN_i
                    # matched_pattern_structure містить оригінальні словники з прапорцями _is_...
                    original_pattern_with_flags = caliber_patterns_config[pattern_index] 
                    
                    value_parts = []
                    # Перевіряємо, чи це патерн, де весь збіг - це один токен, що є значенням
                    if original_pattern_with_flags[0].get("_is_single_value_caliber_token"):
                        text_to_parse = matched_segment.text
                        # Regex для виділення ядра калібру з одного токена (може бути "к.7.62мм" або "12калібру")
                        # Він намагається знайти числову частину, опціональну "x" частину та опціональні "мм"
                        _cal_extract_regex = (
                            f"({NUMERIC_CALIBER_PART_REGEX}"  # Основне число
                            f"(?:[xхXХ×]{SECOND_NUM_PART_REGEX})?"  # Опціонально: x39, x54R
                            f"(?:мм|mm)?)"  # Опціонально: мм
                            f"|({NUMERIC_CALIBER_PART_REGEX}(?:мм|mm)?)" # Або просто число + мм
                            f"|({NUMERIC_CALIBER_PART_REGEX})" # Або просто число
                        )
                        search_res = re.search(_cal_extract_regex, text_to_parse)
                        if search_res:
                            # re.search(...).group(0) поверне весь знайдений підрядок
                            extracted_caliber = search_res.group(0) 
                        else:
                            extracted_caliber = text_to_parse # Fallback
                    else: # Для багатотокенних патернів
                        # Ітеруємо по фактично знайдених токенах у збігу (`matched_segment`)
                        # і збираємо ті, що схожі на частини калібру за їх вмістом.
                        for token_in_segment in matched_segment:
                            token_text = token_in_segment.text
                            token_lower = token_in_segment.lower_
                            
                            is_value_part_token = False
                            if re.fullmatch(NUMERIC_CALIBER_PART_REGEX, token_text):
                                is_value_part_token = True
                            elif token_lower in ["x", "х", "×"]: # З X_SEP_DEF
                                is_value_part_token = True
                            elif re.fullmatch(SECOND_NUM_PART_REGEX, token_text): # З NUM_PART2_DEF
                                is_value_part_token = True
                            elif token_lower in ["мм", "mm"]:   # З UNIT_MM_DEF
                                is_value_part_token = True
                            
                            if is_value_part_token:
                                value_parts.append(token_text)
                        
                        if value_parts:
                            extracted_caliber = " ".join(value_parts).strip()
                        else:
                            extracted_caliber = matched_segment.text 
                
                except Exception as e_extraction:
                    logger.warning(f"Помилка виділення значення калібру з '{matched_segment.text}': {e_extraction}", exc_info=True)
                    extracted_caliber = matched_segment.text 
                
                candidate_matches_for_current_span_cal.append({
                    'full_matched_text': matched_segment.text,
                    'original_segment_start_char': original_segment_start_char,
                    'original_segment_end_char': original_segment_end_char,
                    'extracted_caliber_value': extracted_caliber
                })
            
            if candidate_matches_for_current_span_cal:
                candidate_matches_for_current_span_cal.sort(
                    key=lambda m: (m['original_segment_start_char'], -(m['original_segment_end_char'] - m['original_segment_start_char'])))
                selected_matches_for_this_span_cal = [candidate_matches_for_current_span_cal[0]]
                for current_candidate_data in candidate_matches_for_current_span_cal[1:]:
                    last_added_match_data = selected_matches_for_this_span_cal[-1]
                    if current_candidate_data['original_segment_start_char'] == last_added_match_data['original_segment_start_char']: continue
                    if current_candidate_data['original_segment_start_char'] >= last_added_match_data['original_segment_end_char']:
                        selected_matches_for_this_span_cal.append(current_candidate_data)
                        continue
                    if current_candidate_data['original_segment_end_char'] <= last_added_match_data['original_segment_end_char']: continue
                
                for final_match_data in selected_matches_for_this_span_cal:
                    all_final_caliber_matches.append(WeaponCaliberMatch(
                        entity_text=ent.text, entity_label=ent.label_, entity_start_ch=ent.start_char, entity_end_ch=ent.end_char,
                        matched_caliber_segment_text=final_match_data['full_matched_text'],
                        matched_segment_start_char=final_match_data['original_segment_start_char'],
                        matched_segment_end_char=final_match_data['original_segment_end_char'],
                        extracted_caliber_value=final_match_data['extracted_caliber_value']
                    ))
        results.weapon_calibers = all_final_caliber_matches
        # --- Кінець обробки КАЛІБРІВ ---

        end_time = time.time() 
        processing_duration = end_time - start_time

        if stats_manager: 
            stats_manager.update_stats(doc, processing_duration)

    except Exception as e:
        logger.error(f"Помилка обробки тексту в '/analyze_fabula': '{fabula[:100]}...'. Виняток: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Помилка обробки тексту: {e}")
    
    return results

# --- Ендпоінт для пакетної обробки ---
@app.post("/analyze_fabulas_batch", response_model=FabulaBatchOutput, summary="Analyze a batch of fabulas")
async def analyze_fabulas_batch(batch_input: FabulaBatchInput):
    if nlp is None: 
        logger.error("'/analyze_fabulas_batch' was called before model was loaded")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Модель Spacy не завантажено, спробуйте пізніше") 
    
    batch_results_final: List[FabulaBatchOutputItem] = [] # Використовуємо нове ім'я
    processed_ctr: int = 0
    failed_ctr: int  = 0
    
    for fabula_item_input in batch_input.fabulas: # Змінено ім'я змінної для ясності
        # Для кожного елемента батча ми можемо викликати логіку, схожу на /analyze_fabula,
        # але потрібно адаптувати вхід/вихід або створити допоміжну функцію.
        # Зараз просто імітуємо отримання сутностей, як у вашому коді.
        # В РЕАЛЬНОСТІ ТУТ МАЄ БУТИ ПОВНА ЛОГІКА АНАЛІЗУ ЯК В /analyze_fabula
        
        fabula_text = fabula_item_input.fabula # Отримуємо текст з FabulaInput
        doc_batch = nlp(fabula_text)
        entities_for_item = [(ent.text, ent.label_) for ent in doc_batch.ents]
        
        # ТУТ ПОТРІБНО ДОДАТИ ЛОГІКУ ДЛЯ SN ТА CALIBER, ЯК В /analyze_fabula
        # і заповнити відповідні поля в FabulaBatchOutputItem (серійні номери, калібри)
        # Наприклад:
        # weapon_serials_for_item = process_serial_numbers(doc_batch, serial_number_matcher, ...)
        # weapon_calibers_for_item = process_calibers(doc_batch, caliber_matcher, ...)

        batch_results_final.append(FabulaBatchOutputItem(
            entities=entities_for_item
            # weapon_serial_numbers=weapon_serials_for_item, # Має бути додано
            # weapon_calibers=weapon_calibers_for_item      # Має бути додано
        ))
        processed_ctr += 1
        # Обробка помилок для окремого елемента батча тут не реалізована,
        # але її можна додати, збільшуючи failed_ctr
        
    logger.info(f"Завершено обробку батчу: успішно {processed_ctr}, помилок {failed_ctr}")
    # failed_ctr тут завжди буде 0, якщо немає індивідуальної обробки помилок
    return FabulaBatchOutput(results=batch_results_final, processed_ctr=processed_ctr, failed_ctr=failed_ctr)


# --- Endpoint'и для статистики (залишаються без змін) ---
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

@app.get("/stats/entities_by_label_plot", summary="Get a plot of total weapon entities found by label (PNG)")
async def get_entities_by_label_plot():
    if stats_manager is None: 
        logger.error("'/stats/entities_by_label_plot' called before StatsManager was initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Система статистики ще не готова або недоступна. Спробуйте пізніше.")
    try:
        current_stats = stats_manager.get_stats()
        entities_by_label = current_stats['total_stats'].get('entities_by_label', {})
        if not entities_by_label:
            return JSONResponse(content={"message": "Статистика сутностей за мітками ще не доступна або порожня."},
                                status_code=status.HTTP_404_NOT_FOUND)
        sorted_entities = dict(sorted(entities_by_label.items(), key=lambda item: item[1], reverse=True))
        labels = list(sorted_entities.keys())
        counts = list(sorted_entities.values())
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color='skyblue')
        plt.xlabel("Мітка сутності")
        plt.ylabel("Кількість знайдених сутностей")
        plt.title("Загальна кількість знайденої зброї за мітками")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Помилка при генерації графіка в '/stats/entities_by_label_plot': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Помилка при генерації графіка: {e}")