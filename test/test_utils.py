# -*- coding: utf-8 -*-

"""A module to test how model inputs and outputs are handled."""

import unittest
from array import array

import numpy as np
from emmorphpy import EmMorphPy

from anonymizer import (
    format_classifier_outputs,
    restore_words_and_labels,
    restore_spaces,
    replace_named_entities,
    create_byte_ids,
    split_id_sequence,
    format_decoder_output,
    decode_output,
    get_morph_suffix,
    fix_articles
)
from anonymizer.ling_data_structures import AnnotatedToken


class UtilsTestCase(unittest.TestCase):
    """A test case to test utils."""

    def test_format_classifier_outputs(self) -> None:
        predictions = np.array([[[0.98, 1.2, -0.31], [-1.45, -0.29, 0.07]]])
        tokens = [["Yes", "."]]
        expected_result = [("Yes", 1), (".", 2)]
        result = next(format_classifier_outputs(predictions, tokens))
        self.assertEqual(expected_result, result)  # add assertion here

    def test_restore_words_and_labels(self) -> None:
        inputs = [("My", 0), ("dog", 0), ("Har", 1), ("##ry", 2), ("bark", 0), ("##s", 0)]
        label_map = {0: "O", 1: "B-PER", 2: "I-PER"}
        expected_result = [
            AnnotatedToken("My", ner="O"),
            AnnotatedToken("dog", ner="O"),
            AnnotatedToken("Harry", ner="B-PER"),
            AnnotatedToken("barks", ner="O")
        ]
        result = restore_words_and_labels(inputs, label_map=label_map)
        self.assertEqual(expected_result, result)

    def test_restore_spaces(self) -> None:
        inputs = [
            AnnotatedToken("Hungary"),
            AnnotatedToken("("),
            AnnotatedToken("hu"),
            AnnotatedToken(")"),
            AnnotatedToken("won"),
            AnnotatedToken(".")
        ]
        expected_wsafter = [" ", "", "", " ", "", ""]
        restore_spaces(inputs)
        result_wsafter = [token.wsafter for token in inputs]
        self.assertEqual(expected_wsafter, result_wsafter)

    def test_replace_named_entities(self) -> None:
        name_map = {"PER": [["John", "Taylor", "Smith"], ["John", "Smith"]]}
        input_tokens = [
            AnnotatedToken("Harry", ner="B-PER"),
            AnnotatedToken("lives", ner="O"),
        ]
        expected_result = [
            AnnotatedToken("John", ner="B-PER"),
            AnnotatedToken("Smith", ner="I-PER"),
            AnnotatedToken("lives", ner="O")
        ]
        result = replace_named_entities(
            annotated_tokens=input_tokens,
            replacement_names=name_map
        )
        self.assertEqual(expected_result, result)

    def test_create_byte_ids(self) -> None:
        offset = 3
        sep_id, eos_id = 259, 1
        inputs = [
            AnnotatedToken("Harry", ner="B-PER"),
            AnnotatedToken("barks", "", "O"),
            AnnotatedToken(".", "", "O")
        ]
        expected_result = array("H", (byte_val + offset for byte_val
                                      in "Harry barks.".encode("utf-8")))
        expected_result.insert(5, sep_id)
        expected_result.insert(0, sep_id)
        expected_result.append(eos_id)
        result = create_byte_ids(
            annotated_tokens=inputs,
            offset=offset,
            sep_id=sep_id,
            eos_id=eos_id
        )
        self.assertEqual(expected_result, result)

    def test_split_id_sequence(self) -> None:
        id_sequence = array("H", [259, 10, 100, 259, 35, 128, 1, 259, 10, 101, 259, 1])
        expected_condition = id_sequence[:6]
        expected_generated = id_sequence[7:-1]
        result_condition, result_generated = split_id_sequence(id_sequence, eos_id=1)
        self.assertEqual(expected_condition, result_condition)
        self.assertEqual(expected_generated, result_generated)

    def test_format_decoder_output(self) -> None:
        condition = array("H", [147, 35, 259, 10, 100, 259, 35, 128])
        generated = array("H", [259, 10, 101, 259])
        expected_result = array("H", [147, 35, 10, 101, 35, 128])
        result = format_decoder_output(condition, generated, sep_id=259)
        self.assertEqual(expected_result, result)

    def test_decode_output(self) -> None:
        byte_ids = array("H", [75, 100, 117, 117, 124])
        expected_result = "Harry"
        result = decode_output(byte_ids, offset=3)
        self.assertEqual(expected_result, result)

    def test_get_morph_suffix(self) -> None:
        m = EmMorphPy()
        token = AnnotatedToken(form="vakondok", tag="[/N][Pl][Nom]")
        res = get_morph_suffix(token, m)
        self.assertEqual("-ok", res)

    def test_fix_articles(self) -> None:
        sentence = [
            AnnotatedToken("Az", ner="O"),
            AnnotatedToken("Duna", ner="B-LOC"),
            AnnotatedToken("Magyarországon", ner="B-LOC"),
            AnnotatedToken("folyik", ner="O"),
            AnnotatedToken(".", ner="O")
        ]
        fix_articles(sentence)
        result = " ".join(word.form for word in sentence)
        expected_result = "A Duna Magyarországon folyik ."
        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
