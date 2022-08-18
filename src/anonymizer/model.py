# -*- coding: utf-8 -*-

"""A module for the full anonymizer engine."""

import json
from array import array
from warnings import warn
from types import MappingProxyType
from re import compile as re_compile
from unicodedata import normalize as unicode_normalize
from os import getenv
from os.path import isdir, join as os_path_join
from typing import Iterable, List, Sequence, MutableSequence, Union, Optional

import torch
import purepospy
from emmorphpy import EmMorphPy
from quntoken import tokenize as quntokenize
from numpy import array as np_array
from transformers import AutoTokenizer, AutoModelForTokenClassification, GPT2LMHeadModel

from anonymizer import (
    AnnotatedToken,
    CorrectorArgs,
    format_classifier_outputs,
    restore_words_and_labels,
    restore_spaces,
    replace_named_entities,
    create_byte_ids,
    split_id_sequence,
    format_decoder_output,
    decode_output,
    add_tags,
    fix_articles
)


class Anonymizer:
    """The anonymizer class: it is a tool whose input is a string.
    It replaces named entities in the input with other named entities
    selected from the list. It outputs the modified string.

    The anonymizer uses neural models to recognize named entities and
    the put the new named entities into the forms that are required
    by the context.
    """

    def __init__(
            self, *,
            tokenizer_path: str,
            detector_path: str,
            label_map_path: str,
            mask_map_path: str,
            entity_map_path: Optional[str] = None,
            definite_entity_map_path: Optional[str] = None,
            corrector_args: Optional[CorrectorArgs] = None
    ) -> None:
        """Initialize the anonymizer.

        Args:
            tokenizer_path: Path to the detector tokenizer.
            detector_path: Path to the trained detector model.
            label_map_path: Path to a `json` file that maps the corrector predictions
                to proper NER labels.
            mask_map_path: Path to a `json` file that maps NER categories to tags.
            entity_map_path: Path to a `json` file that maps NER categories to
                lists of named entities with which named entities in the anonymizer
                inputs will be replaced. Only relevant if `corrector_args` is specified.
                Optional.
            definite_entity_map_path: The same as `entity_map_path`, but used only
                when a definite article was detected before a named entity. Only relevant
                if `entity_map_path` and `corrector_args` are specified. Optional.
            corrector_args: The arguments used to initialize the corrector.
                Only relevant if `entity_map_path` is specified. Optional.
        """
        self._hyphen_pattern = re_compile(r"(\s-\s)|(^-\s)")  # A regexp pattern to handle hyphens
        with open(mask_map_path, "rb") as mask_file:
            self._mask_map = json.load(mask_file)
        with open(label_map_path, "rb") as label_file:
            self._label_map = {int(k): v for k, v in json.load(label_file).items()}

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._detector = AutoModelForTokenClassification.from_pretrained(detector_path).eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._detector.to(self._device)

        if corrector_args is not None and entity_map_path is not None:
            with open(entity_map_path, "rb") as entity_file:
                self._entity_map = json.load(entity_file)
            if definite_entity_map_path is not None:
                with open(definite_entity_map_path, "rb") as definite_entity_file:
                    self._definite_entity_map = json.load(definite_entity_file)
            else:
                self._definite_entity_map = None
            self._corrector_args = corrector_args
            self._corrector = GPT2LMHeadModel.from_pretrained(
                self._corrector_args.model_path).eval()
            self._corrector.to(self._device)
        else:
            if corrector_args is not None:
                warn("No entity replacement map was specified for the corrector. "
                     "Falling back to analyzers only.")
            self._corrector_args = None
            self._corrector = None
            self._entity_map = None
            self._definite_entity_map = None

        self._purepos = purepospy.PurePOS(
            os_path_join(next(iter(purepospy.__path__)), "szeged.model"))
        list(self._purepos.tag_sentence(["Ez", "minden", "."]))  # Initialize PurePOS by calling it.
        self._emmorph = EmMorphPy()

    def _get_detector_prediction(
            self,
            text_batch: Union[Sequence[str], str],
            drop_special_tokens: bool = True
    ) -> np_array:
        """Tokenize a string and feed it to the detector.
        The detector input can be a sequence of examples that corresponds to a batch.
        """
        detector_inputs = self._tokenizer(text_batch, padding=True, return_tensors="pt",
                                          return_token_type_ids=False)
        for key, detector_input in detector_inputs.items():
            detector_inputs[key] = detector_input.to(self._device)
        with torch.no_grad():
            detector_predictions = self._detector(**detector_inputs).logits
        if drop_special_tokens:
            detector_predictions = detector_predictions[:, 1:-1, :]
        return detector_predictions.cpu().numpy()

    def _get_corrector_prediction(
            self,
            token_batch: Iterable[List[AnnotatedToken]],
            input_length: Optional[int] = None
    ) -> np_array:
        """Feed a batch of formatted detector outputs to the corrector."""
        byte_ids = [create_byte_ids(
            annotated_tokens=tokens,
            offset=self._corrector_args.offset,
            sep_id=self._corrector_args.sep_id,
            eos_id=self._corrector_args.eos_id
        ) for tokens in token_batch]
        byte_ids = torch.tensor(byte_ids, dtype=torch.int32).to(self._device)
        if input_length is None:
            input_length = self._corrector_args.max_input_length
        return torch.squeeze(self._corrector.generate(
            byte_ids, max_length=input_length * 2)).cpu().numpy()

    def __call__(self, text: str) -> str:
        """Call the model to anonymize the input text."""
        text = self._hyphen_pattern.sub(" \u2014 ", unicode_normalize("NFKC", text))
        result = []
        for sentence in map(str.strip, quntokenize(text, form="spl")):
            tokens = self._tokenizer.tokenize(sentence)
            logits = self._get_detector_prediction(sentence)
            token_seq = next(format_classifier_outputs(logits, [tokens]))
            del logits, tokens
            token_seq = restore_words_and_labels(token_seq, self._label_map)

            if self._corrector_args is not None:
                input_len_bytes = len(list(text.encode(encoding="utf-8")))
                assert input_len_bytes <= self._corrector_args.max_input_length, \
                    f"The input is {input_len_bytes}, the maximum length is " \
                    f"{self._corrector_args.max_input_length}."
                try:
                    res = self._call_corrector(token_seq, input_len_bytes)
                except AssertionError:
                    res = self._call_analyzers(token_seq)
            else:
                res = self._call_analyzers(token_seq)
            result.append(res)
        return " ".join(result)

    def _call_corrector(
            self,
            words: MutableSequence[AnnotatedToken],
            input_len_bytes: int
    ) -> str:
        """Call a neural corrector (GPT) after replacing named entities."""
        words = replace_named_entities(
            annotated_tokens=words,
            replacement_names=self._entity_map,
            definite_names=self._definite_entity_map
        )
        restore_spaces(words)
        fix_articles(words)
        words = self._get_corrector_prediction(
            [words], input_length=input_len_bytes).tolist()
        condition, generated = split_id_sequence(array("H", words),
                                                 eos_id=self._corrector_args.eos_id)
        corrector_output = format_decoder_output(condition, generated,
                                                 sep_id=self._corrector_args.sep_id)
        return decode_output(corrector_output, offset=self._corrector_args.offset)

    def _call_analyzers(self, words: MutableSequence[AnnotatedToken]) -> str:
        """Call `PurePOS` and `EmMorph` if a neural corrector is not available."""
        add_tags(words, analyzer=self._purepos)
        words = replace_named_entities(
            annotated_tokens=words,
            replacement_names=self._mask_map,
            analyzer=self._emmorph
        )
        restore_spaces(words)
        return "".join(word.form + word.wsafter for word in words)

    @property
    def has_corrector(self) -> bool:
        """Check if the anonymizer has a neural corrector part."""
        return self._corrector is not None

    @property
    def corrector_args(self) -> Union[MappingProxyType, None]:
        if self._corrector_args is None:
            return self._corrector_args
        return MappingProxyType(self._corrector_args.__dict__)

    @property
    def entity_map(self) -> Union[MappingProxyType, None]:
        if self._entity_map is None:
            return self._entity_map
        return MappingProxyType(self._entity_map)

    @property
    def definite_entity_map(self) -> Union[MappingProxyType, None]:
        if self._definite_entity_map is None:
            return self._definite_entity_map
        return MappingProxyType(self._definite_entity_map)

    @property
    def mask_map(self) -> MappingProxyType:
        return MappingProxyType(self._mask_map)

    @property
    def label_map(self) -> MappingProxyType:
        return MappingProxyType(self._label_map)


def setup_model(
        *, tokenizer_path: str,
        detector_path: str,
        label_map_path: str,
        mask_map_path: str,
        corrector_path: Optional[str] = None,
        entity_map_path: Optional[str] = None,
        definite_entity_map_path: Optional[str] = None,
        **kwargs
) -> Anonymizer:
    """A function that helps set up the model for the application.

    It decides whether a corrector model is required based on the `USE_CORRECTOR` environment
    variable and the existence of `corrector_path`.

    `corrector_path` is a path to a corrector model directory. The other arguments are the same
    as in the `__init__` method of `Anonymizer`. `kwargs` are keyword arguments for `CorrectorArgs`
    (except `model_path`).
    """
    use_corrector = getenv("USE_CORRECTOR", "false").lower() in {"1", "on", "true", "yes", "y"}
    if use_corrector and corrector_path is not None and isdir(corrector_path):
        corrector_args = CorrectorArgs(corrector_path, **kwargs)
    else:
        corrector_args = None
    return Anonymizer(
        tokenizer_path=tokenizer_path,
        detector_path=detector_path,
        label_map_path=label_map_path,
        entity_map_path=entity_map_path,
        definite_entity_map_path=definite_entity_map_path,
        mask_map_path=mask_map_path,
        corrector_args=corrector_args
    )
