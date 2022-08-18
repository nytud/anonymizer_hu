# -*- coding: utf-8 -*-

"""A module to define linguistic data structures."""

import json
from dataclasses import dataclass, field
from typing import Optional, FrozenSet, Union, IO


@dataclass
class AnnotatedToken:
    """A dataclass for tokens with NER tags."""
    form: str = field(metadata={"help": "The token form as a string."})
    wsafter: str = field(
        default=" ",
        metadata={
            "help": "The string (usually a space character or the empty string) "
                    "that follows the given token form."
        }
    )
    ner: Optional[str] = field(default=None, metadata={"help": "A NER tag."})
    tag: Optional[str] = field(default=None, metadata={"help": "Morphological tag(s) as a string."})


@dataclass(frozen=True)
class FuncWords:
    """A dataclass for sets of function words grouped by POS."""
    conjunctions: FrozenSet[str] = field(
        default=frozenset(),
        metadata={"help": "A set of conjunctions."}
    )
    prepositions: FrozenSet[str] = field(
        default=frozenset(),
        metadata={"help": "A set of prepositions."}
    )
    postpositions: FrozenSet[str] = field(
        default=frozenset(),
        metadata={"help": "A set of postpositions."}
    )
    undef_articles: FrozenSet[str] = field(
        default=frozenset(),
        metadata={"help": "A set of undefinite articles."}
    )
    def_articles: FrozenSet[str] = field(
        default=frozenset(),
        metadata={"help": "A set of definite articles."}
    )
    pre_vowel_def_article: Optional[str] = field(
        default=None,
        metadata={"help": "The definite article used before vowels. This is relevant in Hungarian."}
    )
    pre_cons_def_article: Optional[str] = field(
        default=None,
        metadata={"help": "The definite article used before consonants. This is relevant in Hungarian."}
    )

    @classmethod
    def from_json(cls, json_file: Union[IO, str]):
        """Construct an object from a `json` file."""
        if isinstance(json_file, str):
            with open(json_file, "rb") as jf:
                data = json.load(jf)
        else:
            data = json.load(json_file)
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = frozenset(v)
        return cls(**data)
