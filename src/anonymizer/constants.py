# -*- coding: utf-8 -*-

"""A module to define constants."""

from os.path import dirname, abspath, join as os_path_join

from anonymizer.ling_data_structures import FuncWords

# Paths
SOURCES_ROOT = dirname(dirname(abspath(__file__)))
PROJECT_ROOT = dirname(SOURCES_ROOT)
MODEL_DIR = os_path_join(PROJECT_ROOT, "models")
DETECTOR = os_path_join(MODEL_DIR, "detector")
CORRECTOR = os_path_join(MODEL_DIR, "corrector")
MASK_MAP = os_path_join(MODEL_DIR, "mask_map.json")
ENTITY_MAP = os_path_join(MODEL_DIR, "entity_map.json")
DEFINITE_ENTITY_MAP = os_path_join(MODEL_DIR, "definite_entity_map.json")
LABEL_MAP = os_path_join(MODEL_DIR, "label_map.json")
TOKENIZER = "SZTAKI-HLT/hubert-base-cc"

# Linguistic constants
LEFT_PUNCT = frozenset({"(", "[", "{"})
RIGHT_PUNCT = frozenset({")", "]", "}", ".", ",", ";", ":", "!", "?"})
VOWELS = frozenset("AÁEÉIÍOÓÖŐUÚÜŰaáeéiíoóöőuúüű")
FUNC_WORDS = FuncWords(
    conjunctions=frozenset({"és", "s", "vagy", "illetve", "valamint"}),
    def_articles=frozenset({"a", "az"}),
    pre_vowel_def_article="az",
    pre_cons_def_article="a"
)

#Models
MAX_INPUT_LENGTH = 450
