from fastapi import FastAPI
from fastapi.testclient import TestClient
from .main import app


client =  TestClient(app)

def test_ner_aviability():
    response  = client.post("/ner/", json={"text":""})
    assert response.status_code  == 200

def test_ner_bad_params():
    response = client.post("/ner/", json={"":""})
    assert response.status_code == 422

def test_docs_locally():
    response = client.get("/docs")
    assert response.status_code == 200

def test_ner_response_fields():
    response = client.post("/ner/", json={"text":"Автомат АК-74 кб 5.45мм, 1990 р.в. серійний номерЖ 123456"})
    assert response.status_code == 200