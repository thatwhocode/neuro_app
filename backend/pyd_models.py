from pydantic import BaseModel
from typing import List, Tuple, Optional


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
class WeaponCaliberMatch(BaseModel):
    entity_text: str 
    entity_label : str 
    entity_start_ch: int 
    entity_end_ch: int
    matched_caliber_segment_text: str
    matched_segment_start_char: int 
    matched_segment_end_char: int
    extracted_caliber_value: Optional[str] = None
class FabulaOutput(BaseModel):
    entities: List[Tuple[str, str]] = []
    weapon_serial_numbers :List[WeaponSerialNumber] =[]
    weapon_calibers: List[WeaponCaliberMatch] = []
    
class FabulaBatchInput(BaseModel):
    fabulas : List[FabulaInput]


class FabulaBatchOutputItem(BaseModel): 
    entities: List[Tuple[str, str]] = []
     
class FabulaBatchOutput(BaseModel):
    results: List[FabulaBatchOutputItem] = []
    processed_ctr: int
    failed_ctr: int = 0