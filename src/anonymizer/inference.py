#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Anonymizer inference script."""

from os.path import join as os_path_join

from anonymizer import Anonymizer, CorrectorArgs

PREF = "/home/nytud/Projects/corp/anonymizer"
DETECTOR = os_path_join(PREF, "torch_nerbert")
CORRECTOR = os_path_join(PREF, "torch_nergpt")
ENTITY_MAP = os_path_join(PREF, "entity_map.json")
DEFINITE_ENTITY_MAP = os_path_join(PREF, "definite_entity_map.json")
# ENTITY_MAP = os_path_join(PREF, "mask_map.json")
LABEL_MAP = os_path_join(PREF, "label_map.json")
TOKENIZER = "SZTAKI-HLT/hubert-base-cc"


def main() -> None:
    """Run the model in an infinite loop."""
    model = Anonymizer(
        tokenizer_path=TOKENIZER,
        detector_path=DETECTOR,
        label_map_path=LABEL_MAP,
        entity_map_path=ENTITY_MAP,
        definite_entity_map_path=DEFINITE_ENTITY_MAP,
        corrector_args=CorrectorArgs(CORRECTOR)
    )
    while True:
        text = input("Type you input: ")
        if text.strip().lower() == "exit":
            break
        result = model(text)
        print(result)


if __name__ == "__main__":
    main()
