from fastapi import FastAPI
from fastapi.testclient import TestClient
from .main import app


client =  TestClient(app)

def test_ner_aviability():
    response  = client.post("/ner/", json={"text":""})
    assert response.status_code  == 200