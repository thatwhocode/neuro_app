from pydantic import BaseModel
from typing import List, Tuple


class FabulaInput(BaseModel):
    fabula: str

class FabulaOutput(BaseModel):
    entities: List[Tuple[str, str]] = []


class FabulaBatchInput(BaseModel):
    fabulas : List[FabulaInput]


class FabulaBatchOutputItem(BaseModel): 
    entities: List[Tuple[str, str]] = []
     
class FabulaBatchOutput(BaseModel):
    results: List[FabulaBatchOutputItem] = []
    processed_ctr: int
    failed_ctr: int = 0