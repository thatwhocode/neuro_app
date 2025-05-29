from pydantic import BaseModel
from typing import List, Tuple


class FabulaInput(BaseModel):
    fabula: str
class WeaponSerialNumber(BaseModel):
    entity_text: str 
    entity_label : str 
    entity_start_ch: int 
    entity_end_ch: int
    serial_number : str
    matched_weapon_segment_text: str 
    matched_segment_start_char: int
    matched_segment_end_char : int
class WeaponCaliber(BaseModel):
    entity_text: str 
    entity_label : str 
    entity_start_ch: int 
    entity_end_ch: int
    weapon_caliber: str
    matched_caliber_segment_text: str
    matched_segment_start: str 
    mathced_segment_end: str
class FabulaOutput(BaseModel):
    entities: List[Tuple[str, str]] = []
    weapon_serial_numbers :List[WeaponSerialNumber] =[]

class FabulaBatchInput(BaseModel):
    fabulas : List[FabulaInput]


class FabulaBatchOutputItem(BaseModel): 
    entities: List[Tuple[str, str]] = []
     
class FabulaBatchOutput(BaseModel):
    results: List[FabulaBatchOutputItem] = []
    processed_ctr: int
    failed_ctr: int = 0