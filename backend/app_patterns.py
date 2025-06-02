
from app_config import(
    sn_token_single_def, sn_token_prefix_def, sn_token_suffix_def, sn_token_single_letter_prefix_def,
    markers_text_lower_sn, markers_symbols_sn,
    NUM_PART1_DEF, X_SEP_DEF,NUM_PART2_DEF, MARKER_SUFFIX_KALIBRU_DEF, OPTIONAL_SEPARATOR_DEF,
    K_NUM_MM_SINGLE_TOKEN_REGEX, K_NUMXNUM_MM_SINGLE_TOKEN_REGEX, NUM_KALIBRU_SINGLE_TOKEN_REGEX,
    NUMXNUM_KALIBRU_SINGLE_TOKEN_REGEX,MARKER_KCAL_DEF, MARKER_K_ABBR_DEF, UNIT_MM_DEF)
def generate_weapons_number_patterns():
    weapon_number_patterns_config  = []
    for marker_sym in markers_symbols_sn:
        weapon_number_patterns_config.append([{"TEXT": marker_sym}, {"TEXT": ":", "OP": "?"}, sn_token_single_def])
        weapon_number_patterns_config.append([{"TEXT": marker_sym}, sn_token_single_def])
        weapon_number_patterns_config.append([{"TEXT": marker_sym}, {"TEXT": ":", "OP": "?"}, sn_token_prefix_def, sn_token_suffix_def])
        weapon_number_patterns_config.append([{"TEXT": marker_sym}, sn_token_prefix_def, sn_token_suffix_def])
        weapon_number_patterns_config.append([{"TEXT": marker_sym}, {"TEXT": ":", "OP": "?"}, sn_token_single_letter_prefix_def, sn_token_suffix_def])
        weapon_number_patterns_config.append([{"TEXT": marker_sym}, sn_token_single_letter_prefix_def, sn_token_suffix_def])
    for marker_txt in markers_text_lower_sn:
        weapon_number_patterns_config.append([{"LOWER": marker_txt}, {"TEXT": {"IN": [":", "."]}, "OP": "?"}, sn_token_single_def])
        weapon_number_patterns_config.append([{"LOWER": marker_txt}, sn_token_single_def])
        weapon_number_patterns_config.append([{"LOWER": marker_txt}, {"TEXT": {"IN": [":", "."]}, "OP": "?"}, sn_token_prefix_def, sn_token_suffix_def])
        weapon_number_patterns_config.append([{"LOWER": marker_txt}, sn_token_prefix_def, sn_token_suffix_def])
        weapon_number_patterns_config.append([{"LOWER": marker_txt}, {"TEXT": {"IN": [":", "."]}, "OP": "?"}, sn_token_single_letter_prefix_def, sn_token_suffix_def])
        weapon_number_patterns_config.append([{"LOWER": marker_txt}, sn_token_single_letter_prefix_def, sn_token_suffix_def])
        if marker_txt in ["ном", "н"]:
            weapon_number_patterns_config.append([{"LOWER": marker_txt}, {"TEXT": "."}, sn_token_single_def])
            weapon_number_patterns_config.append([{"LOWER": marker_txt}, {"TEXT": "."}, sn_token_prefix_def, sn_token_suffix_def])
            weapon_number_patterns_config.append([{"LOWER": marker_txt}, {"TEXT": "."}, sn_token_single_letter_prefix_def, sn_token_suffix_def])
    return weapon_number_patterns_config


def generate_caliber_patterns():
    
    caliber_patterns_config = []
# Група 1: Маркер ("калібр", "кал", "кл") ПЕРЕД значенням
    caliber_patterns_config.append([MARKER_KCAL_DEF, OPTIONAL_SEPARATOR_DEF, NUM_PART1_DEF, UNIT_MM_DEF])
    caliber_patterns_config.append([MARKER_KCAL_DEF, OPTIONAL_SEPARATOR_DEF, NUM_PART1_DEF, X_SEP_DEF, NUM_PART2_DEF, UNIT_MM_DEF])
# Група 2: Маркер "к" ПЕРЕД значенням
    caliber_patterns_config.append([MARKER_K_ABBR_DEF, OPTIONAL_SEPARATOR_DEF, NUM_PART1_DEF, UNIT_MM_DEF])
    caliber_patterns_config.append([MARKER_K_ABBR_DEF, OPTIONAL_SEPARATOR_DEF, NUM_PART1_DEF, X_SEP_DEF, NUM_PART2_DEF, UNIT_MM_DEF])
# Патерни для злитого написання (якщо токенізується як одне слово)

    caliber_patterns_config.append([{"TEXT": {"REGEX": K_NUM_MM_SINGLE_TOKEN_REGEX}, "_is_single_value_caliber_token": True}])

    caliber_patterns_config.append([{"TEXT": {"REGEX": K_NUMXNUM_MM_SINGLE_TOKEN_REGEX}, "_is_single_value_caliber_token": True}])
# Група 3: Значення ПЕРЕД маркером "калібру"
    caliber_patterns_config.append([NUM_PART1_DEF, MARKER_SUFFIX_KALIBRU_DEF])
    caliber_patterns_config.append([NUM_PART1_DEF, X_SEP_DEF, NUM_PART2_DEF, MARKER_SUFFIX_KALIBRU_DEF])
# Патерни для злитого написання

    caliber_patterns_config.append([{"TEXT": {"REGEX": NUM_KALIBRU_SINGLE_TOKEN_REGEX}, "_is_single_value_caliber_token": True}])

    caliber_patterns_config.append([{"TEXT": {"REGEX": NUMXNUM_KALIBRU_SINGLE_TOKEN_REGEX}, "_is_single_value_caliber_token": True}])
    return caliber_patterns_config


weapon_number_patterns_config = generate_weapons_number_patterns()
caliber_patterns_config  = generate_caliber_patterns()

