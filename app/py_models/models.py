from pydantic import BaseModel, Field
from typing import List
class TextInput(BaseModel):
    text: str

class Entity(BaseModel):
    entity_group: str = Field(..., description="Type of entity")
    score: float  = Field(..., description="Level of awarness")
    word: str = Field(..., description="Text of the entity")
    start: int = Field(..., description="Starting symbol of entity")
    end: int = Field(..., description="End char of entity")

class NEROutput(BaseModel):
    entities: List[Entity]