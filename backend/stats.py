from datetime import date, datetime 
from typing import Dict, Any
import logging
from pathlib import Path
import json as js 
import io
import spacy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATS_FILE = "api_processing_stats.json" 

class StatsManager:
    def __init__(self, stats_file: str = STATS_FILE):
        self._stats_file = Path(stats_file)
        

        loaded_stats = self._load_from_file()

        self._total_processed = loaded_stats.get("total_fabulas_processed", 0)
        self._total_entities_found = loaded_stats.get("total_entities_found", 0)
        self._entities_by_label = loaded_stats.get("entities_by_label", {})
        if not isinstance(self._entities_by_label, dict): self._entities_by_label = {}

        self._total_processing_time_seconds = loaded_stats.get("total_processing_time_seconds", 0.0)

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