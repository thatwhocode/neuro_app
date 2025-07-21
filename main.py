from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from models import  TextInput, Entity, NEROutput

MODEL_PATH = "./final_ner_model" # Переконайтеся, що це правильний шлях!

# --- 1. Завантаження моделі та токенізатора (виконується один раз при запуску додатку) ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    # Перемістіть модель на GPU, якщо доступно
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Переведіть модель у режим оцінки
    print(f"Модель завантажена на {device}.")
except Exception as e:
    print(f"Помилка при завантаженні моделі: {e}")
    print("Переконайтеся, що шлях до моделі правильний і модель була збережена коректно.")
    # Можна вийти з програми або обробити помилку інакше
    exit()

# --- 2. Ініціалізація FastAPI додатку ---
app = FastAPI(
    title="NER Model API",
    description="API для розпізнавання іменованих сутностей (NER) за допомогою тонко налаштованої моделі.",
    version="1.0.0",
)

# --- 4. Функція для передбачення NER ---
def predict_ner(text: str):
    # Вхідні дані для токенізатора
    tokenization_kwargs = {
        "text": text,
        "return_tensors": "pt",
        "truncation": True,
        "padding": True,
        "return_offsets_mapping": True # Це те, що ми хочемо зберегти
    }
    
    # Виконуємо токенізацію
    inputs_with_offsets = tokenizer(**tokenization_kwargs).to(device)
    
    # --- ВАЖЛИВЕ ВИПРАВЛЕННЯ ТУТ ---
    # Витягуємо offset_mapping та видаляємо його зі словника, який передається моделі
    offset_mapping = inputs_with_offsets.pop("offset_mapping").squeeze().tolist()
    
    # Тепер `inputs_with_offsets` містить лише те, що модель очікує (input_ids, attention_mask тощо)
    inputs = inputs_with_offsets
    # --- КІНЕЦЬ ВИПРАВЛЕННЯ ---

    with torch.no_grad():
        outputs = model(**inputs) # Тепер тут не буде `offset_mapping`

    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    
    # Якщо predictions - це просто int (для одного токена), перетворимо на список
    if not isinstance(predictions, list):
        predictions = [predictions]

    # Якщо offset_mapping - це список списків, а не список кортежів (для одного токена), перетворимо
    if not isinstance(offset_mapping[0], list) and not isinstance(offset_mapping[0], tuple): # Додав перевірку на tuple, бо може бути [(0,0)]
        offset_mapping = [offset_mapping] # Обробка випадку з одним токеном/реченням

    # tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist()) # Не використовується, можна прибрати

    entities = []
    current_entity_text = ""
    current_entity_label = None
    current_entity_start_char = -1

    # Забезпечте, що ітерація відбувається тільки для відповідних даних
    # Деякі спеціальні токени (як CLS, SEP, PAD) можуть мати offset_mapping (0,0)
    # або не мати їх в деяких токенізаторах.
    # Продовжуємо обробку токенів і їхніх передбачень
    for i in range(len(predictions)):
        pred_id = predictions[i]
        label = model.config.id2label[pred_id]
        
        # Переконаємося, що ми не виходимо за межі offset_mapping
        if i >= len(offset_mapping):
            continue # Пропускаємо, якщо offset_mapping коротший

        start_char, end_char = offset_mapping[i]

        # Пропускаємо спеціальні токени або токени з нульовим діапазоном (часто PAD)
        if start_char == 0 and end_char == 0 and i != 0: # i!=0, щоб не пропустити перший токен CLS, який може мати (0,0)
             if current_entity_label: # Якщо була незакрита сутність, додаємо її
                entities.append(Entity(
                    text=text[current_entity_start_char:current_entity_end_char], # Використовуємо збережені char індекси
                    label=current_entity_label.replace("B-", "").replace("I-", ""),
                    start_char=current_entity_start_char,
                    end_char=current_entity_end_char
                ))
                current_entity_text = ""
                current_entity_label = None
                current_entity_start_char = -1
                current_entity_end_char = -1
             continue

        # Логіка IOB2
        if label.startswith("B-"):
            if current_entity_label: # Закінчуємо попередню сутність, якщо вона була
                entities.append(Entity(
                    text=text[current_entity_start_char:current_entity_end_char],
                    label=current_entity_label.replace("B-", "").replace("I-", ""),
                    start_char=current_entity_start_char,
                    end_char=current_entity_end_char
                ))
            current_entity_start_char = start_char
            current_entity_end_char = end_char
            current_entity_label = label
        elif label.startswith("I-") and current_entity_label:
            expected_label_type = current_entity_label.split('-')[1]
            if label.split('-')[1] == expected_label_type:
                current_entity_end_char = end_char # Розширюємо сутність
            else: # Неузгоджений I-токен, починаємо нову сутність або ігноруємо
                if current_entity_label:
                    entities.append(Entity(
                        text=text[current_entity_start_char:current_entity_end_char],
                        label=current_entity_label.replace("B-", "").replace("I-", ""),
                        start_char=current_entity_start_char,
                        end_char=current_entity_end_char
                    ))
                current_entity_start_char = start_char
                current_entity_end_char = end_char
                current_entity_label = label # Починаємо нову сутність з цього "I-"
        else: # O-токен
            if current_entity_label:
                entities.append(Entity(
                    text=text[current_entity_start_char:current_entity_end_char],
                    label=current_entity_label.replace("B-", "").replace("I-", ""),
                    start_char=current_entity_start_char,
                    end_char=current_entity_end_char
                ))
            current_entity_start_char = -1
            current_entity_end_char = -1
            current_entity_label = None
    
    # Додаємо останню сутність, якщо вона була незакрита після циклу
    if current_entity_label:
        entities.append(Entity(
            text=text[current_entity_start_char:current_entity_end_char],
            label=current_entity_label.replace("B-", "").replace("I-", ""),
            start_char=current_entity_start_char,
            end_char=current_entity_end_char # Останній end_char
        ))
    
    # Важливо: відфільтруйте потенційні порожні або None сутності, якщо вони з'явилися
    final_entities = [ent for ent in entities if ent.text.strip() and ent.start_char != -1 and ent.end_char != -1]
    return final_entities


# --- 5. Визначення ендпоінту API ---
@app.post("/ner/", response_model=NEROutput, summary="Розпізнавання іменованих сутностей")
async def get_ner_entities(input_data: TextInput):
    """
    Розпізнає іменовані сутності (NER) у наданому тексті.
    """
    entities = predict_ner(input_data.text)
    return NEROutput(entities=entities)

