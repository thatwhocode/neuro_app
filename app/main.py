from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict

from transformers.pipelines import pipeline
import torch
from .py_models.models import  TextInput, Entity, NEROutput
import uvicorn
from starlette.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
MODEL_PATH = "app/final_ner_model/" 
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only = True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only = True)
    ner_pipeline = pipeline("token-classification", model= model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else  -1)
except Exception as e:
    print(f"Помилка при завантаженні моделі: {e}")
    print("Переконайтеся, що шлях до моделі правильний і модель була збережена коректно.")
# --- 2. Ініціалізація FastAPI додатку ---
app = FastAPI(
    docs_url=None,
    redoc_url= None,
    title="NER Model API",
    description="API для розпізнавання іменованих сутностей (NER) за допомогою тонко налаштованої моделі.",
    version="1.0.0",
)
app.mount("/static/", StaticFiles(directory="app/static/swagger-ui"), name="static_swagger_ui_files")

# --- 4. Функція для передбачення NER ---
def predict_ner(text: str)->List[Dict]:
    result = ner_pipeline(text)
    return result

# --- 5. Визначення ендпоінту API ---
@app.post("/ner/", response_model=NEROutput, summary="Розпізнавання іменованих сутностей")
async def get_ner_entities(input_data: TextInput) ->NEROutput:
    """
    Розпізнає іменовані сутності (NER) у наданому тексті.
    """
    data = predict_ner(input_data.text)
    return NEROutput(entities=data)


@app.get("/docs", response_class=HTMLResponse)
async def serve_local_files_ui():
    index_html_path = "app/static/swagger-ui/index.html"
    try: 
        with open(index_html_path, "r", encoding='utf-8') as f:
            html_content= f.read()
            return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1> Swagger UI file index.html not found!</h1>", status_code= 404)
    

if __name__ == "main":
    uvicorn.run("main:app", host ="0.0.0.0", port=5000, log_level="info")