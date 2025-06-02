import os 


MODEL_PATH  = os.getenv("MODEL_PATH", ".app/model-best") 
STATS_FILE = "api_processing_stats.json"



SERIAL_TOKEN_REGEX = r"^[A-Za-zА-Яа-яЇїІіЄєҐґ0-9](?:[A-Za-zА-Яа-яЇїІіЄєҐґ0-9./-]{0,23}[A-Za-zА-Яа-яЇїІіЄєҐґ0-9])?$|^[A-Za-zА-Яа-яЇїІіЄєҐґ0-9]{2,25}$"
SERIAL_PREFIX_REGEX = r"^[A-Za-zА-Яа-яЇїІіЄєҐґ]{2,7}$"
SERIAL_SUFFIX_REGEX = r"^(?=[A-Za-zА-Яа-яЇїІіЄєҐґ0-9./-]*[0-9])[A-Za-zА-Яа-яЇїІіЄєҐґ0-9./-]{2,20}$|^[0-9./-]{2,20}$"
SINGLE_LETTER_SERIAL_PREFIX_REGEX = r"^[A-Za-zА-Яа-яЇїІіЄєҐґ]$"

sn_token_single_letter_prefix_def = {"TEXT": {"REGEX": SINGLE_LETTER_SERIAL_PREFIX_REGEX}}
sn_token_single_def = {"TEXT": {"REGEX": SERIAL_TOKEN_REGEX}}
sn_token_prefix_def = {"TEXT": {"REGEX": SERIAL_PREFIX_REGEX}}
sn_token_suffix_def = {"TEXT": {"REGEX": SERIAL_SUFFIX_REGEX}}

markers_text_lower_sn = ["номер", "ном", "н", "маркування"]
markers_symbols_sn = ["№", "#"] 

# --- Регулярні вирази та визначення для КАЛІБРІВ ---
NUMERIC_CALIBER_PART_REGEX = r"(?:\d{1,2}(?:[.,]\d{1,3})?|\.\d{2,3}|(?:4|8|10|12|16|20|24|28|32|36))"
SECOND_NUM_PART_REGEX = r"\d{2,3}[A-Za-zА-Яа-я]?" # Для частини типу "39" або "54R"


NUM_PART1_DEF = {"TEXT": {"REGEX": NUMERIC_CALIBER_PART_REGEX}, "_is_value_caliber": True}
X_SEP_DEF = {"LOWER": {"IN": ["x", "х", "×"]}, "_is_value_caliber": True}
NUM_PART2_DEF = {"TEXT": {"REGEX": SECOND_NUM_PART_REGEX}, "_is_value_caliber": True}
UNIT_MM_DEF = {"LOWER": {"IN": ["мм", "mm"]}, "OP": "?", "_is_value_caliber": True}

MARKER_KCAL_DEF = {"LOWER": {"IN": ["калібр", "кал", "кл"]}, "_is_marker_caliber": True}
MARKER_K_ABBR_DEF = {"LOWER": "к", "_is_marker_caliber": True}
MARKER_SUFFIX_KALIBRU_DEF = {"LOWER": "калібру", "_is_marker_caliber": True}
OPTIONAL_SEPARATOR_DEF = {"TEXT": {"IN": [".", ":", "-"]}, "OP": "?", "_is_marker_caliber": True}


K_NUM_MM_SINGLE_TOKEN_REGEX = r"^[кК][.-]?" + NUMERIC_CALIBER_PART_REGEX + r"(?:мм|mm)?$"
K_NUMXNUM_MM_SINGLE_TOKEN_REGEX = r"^[кК][.-]?" + NUMERIC_CALIBER_PART_REGEX + r"[xхXХ×]" + SECOND_NUM_PART_REGEX + r"(?:мм|mm)?$"


NUM_KALIBRU_SINGLE_TOKEN_REGEX = r"^" + NUMERIC_CALIBER_PART_REGEX + r"[Кк][Аа][Лл][Іі][Бб][Рр][Уу]$"
NUMXNUM_KALIBRU_SINGLE_TOKEN_REGEX = r"^" + NUMERIC_CALIBER_PART_REGEX + r"[xхXХ×]" + SECOND_NUM_PART_REGEX + r"[Кк][Аа][Лл][Іі][Бб][Рр][Уу]$"


CONTEXT_WINDOW_SIZE = 25 