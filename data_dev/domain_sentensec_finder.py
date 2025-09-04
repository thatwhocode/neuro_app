import spacy
from spacy.matcher import Matcher

nlp = spacy.load("uk_core_news_sm")
global_matcher = Matcher(nlp.vocab) 

weapons = [
    "пістолет", "револьвер", "рушниця", "карабін",
    "гвинтівка", "автомат", "кулемет", "обріз", "гранатомет", "міномет",
    "вибухівка", "зенітна установка", "гармата", "спеціальний засіб"
]

all_patterns = []
for weapon_phrase in weapons:

    doc_phrase = nlp(weapon_phrase)

    token_pattern = [{"LEMMA": token.lemma_.lower()}
                     for token in doc_phrase if not token.is_space and not token.is_punct]
    
    if token_pattern: 
        all_patterns.append(token_pattern)


if all_patterns:
    global_matcher.add("WEAPON", all_patterns)
    print(f"Matcher ініціалізовано з {len(all_patterns)} патернами.")
else:
    print("Попередження: Не вдалося створити патерни для Matcher. Перевірте список зброї.")

###########################################end of init#################

def spacy_check_for_weapon(text_to_check: str) -> bool:
    """
    Перевіряє, чи містить текст згадку зброї за допомогою global_matcher.
    """
    doc = nlp(text_to_check)
    matches = global_matcher(doc)
    
    if matches:
        return True
    else:
        return False

def process_large_file(input_path: str, output_path: str):
    """
    Читає великий текстовий файл рядок за рядком,
    перевіряє кожен рядок на наявність згадок зброї
    та записує знайдені рядки у вихідний файл.
    """
    
    try:
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:
            
            line_count = 0
            found_mentions_count = 0
            
            # Ітеруємо по файлу рядок за рядком для ефективної роботи з великими файлами
            for line in infile:
                line_count += 1
                
                # Прибираємо зайві пробіли та символи нового рядка
                cleaned_line = line.strip()
                
                # Перевіряємо рядок на наявність згадок зброї
                # Додаємо перевірку, щоб не обробляти порожні рядки
                if cleaned_line and spacy_check_for_weapon(cleaned_line):
                    outfile.write(cleaned_line + "\n") # Правильна конкатенація
                    found_mentions_count += 1
                
                # Виводимо прогрес (опціонально)
                if line_count % 10000 == 0:
                    print(f"Оброблено {line_count} рядків. Знайдено {found_mentions_count} згадувань.")
                    
    except FileNotFoundError:
        print(f"Помилка: Файл '{input_path}' не знайдено.")
    except UnicodeDecodeError:
        print(f"Помилка кодування: Не вдалося декодувати файл '{input_path}' з використанням 'utf-8'. "
              f"Спробуйте інше кодування (наприклад, 'cp1251') або перевірте файл на пошкодження.")
    except Exception as e:
        print(f"Виникла несподівана помилка під час обробки файлу: {e}")

    print(f"\nОбробка завершена.")
    print(f"Всього оброблено рядків: {line_count}")
    print(f"Всього знайдено та записано згадувань зброї: {found_mentions_count}")
    print(f"Результати збережено у файлі: '{output_path}'")

input_file = 'source/input_data.txt'
output_file = 'source/output_data_corpus.txt'
process_large_file(input_file, output_file)