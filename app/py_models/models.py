from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int

class NEROutput(BaseModel):
    entities: list[Entity]