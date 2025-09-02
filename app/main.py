from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.staticfiles import StaticFiles

from transformers import (AutoTokenizer,
                          AutoModelForTokenClassification,
                          AutoModelForSequenceClassification
                          )
from transformers.pipelines import pipeline

import torch

from typing import List, Dict, Union

from .py_models.models import TextInput, NEROutput

import uvicorn

NER_MODEL_PATH = "app/ner_model/"
CLASSIFICATION_MODEL_PATH = "app/classification_model/"


try:
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH,
                                                  local_files_only=True)
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH,
                                                                local_files_only=True)
    class_tokenizer = AutoTokenizer.from_pretrained(CLASSIFICATION_MODEL_PATH,
                                                     local_files_only=True)
    class_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFICATION_MODEL_PATH,
                            local_files_only=True, num_labels=1)

    ner_pipeline = pipeline("token-classification", 
                            model= ner_model,
                            tokenizer=ner_tokenizer,
                            aggregation_strategy="simple",
                            device=0 if torch.cuda.is_available() else  -1)    
    classification_pipeline = pipeline("text-classification",tokenizer=class_tokenizer, model=class_model,return_all_scores = False,
                               device=0 if torch.cuda.is_available() else  -1, truncation = True)   
    print("Pipelines loaded successfully.")
except Exception as e:
    print(f"Помилка при завантаженні  моделі: {e}")
    print("Переконайтеся, що шлях до  моделі правильний і модель була збережена коректно.")




#-- 2. Ініціалізація FastAPI додатку ---
app = FastAPI(
    docs_url=None,
    redoc_url= None,
    title="NER Model API",
    description="API для розпізнавання іменованих сутностей (NER) за допомогою тонко налаштованої моделі.",
    version="1.0.0",
)
app.mount("/app/static/swagger-ui", StaticFiles(directory="app/static/swagger-ui/"), name="static-swagger-ui")

def binary_weapon_classification(input_data: TextInput)->float:
    class_result = classification_pipeline(input_data.text)
    return class_result[0]['score']


def predict_ner(text: str)->List[Dict]:
    result = ner_pipeline(text)
    return result


# --- 5. Визначення ендпоінту API ---
@app.post("/ner/", response_model=NEROutput, summary="Розпізнавання іменованих сутностей")
async def get_ner_entities(input_data: TextInput) ->NEROutput:
    """
    Розпізнає іменовані сутності (NER) у наданому тексті.
    """
    clasified= binary_weapon_classification(input_data)
    if clasified > 0.5:
        data = predict_ner(input_data.text)
        return NEROutput(entities=data)
    else: 
        return NEROutput(entities=[])

@app.get("/docs", response_class=HTMLResponse)
async def serve_local_files_ui():
    return get_swagger_ui_html(openapi_url=app.openapi_url,
                               title=app.title+ "Swagger UI", 
                               swagger_css_url="app/static/swagger-ui/swagger-ui.css",
                               swagger_js_url="app/static/swagger-ui/swagger-ui-bundle.js",
                               swagger_favicon_url="app/static/swagger-ui/favicon-32x32.png"
                               )
    

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)