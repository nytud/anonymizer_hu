# -*- coding: utf-8 -*-

"""A module for utils that help format the model inputs and outputs."""

import re
from array import array
from random import choice as rand_choice
from dataclasses import dataclass, field
from itertools import islice, tee, chain
from typing import (
    Optional, List, Tuple, Generator, Iterable,
    Mapping, MutableSequence, Sequence
)

import numpy as np
from purepospy import PurePOS
from emmorphpy import EmMorphPy

from anonymizer.ling_data_structures import AnnotatedToken
from anonymizer.constants import (
    LEFT_PUNCT,
    RIGHT_PUNCT,
    VOWELS,
    FUNC_WORDS
)


@dataclass(frozen=True)
class CorrectorArgs:
    """A dataclass for the initialization arguments of a GPT-based corrector."""
    model_path: str = field(metadata={"help": "Path to the corrector model file."})
    offset: int = field(
        default=3,
        metadata={
            "help": "A value by which string byte IDs in the corrector input are shifted."
        }
    )
    sep_id: int = field(
        default=259,
        metadata={
            "help": "The ID of the token that surrounds the elements in the input that need to be corrected."
        }
    )
    eos_id: int = field(
        default=1,
        metadata={"help": "The ID of the token that indicates sequence end."}
    )
    max_input_length: int = field(
        default=450,
        metadata={"help": "The maximal input sequence length in bytes."}
    )


def get_ngrams(sequence: Iterable, n: int = 2) -> Iterable:
    """A helper function to get n-grams (by default, bigrams) from a sequence."""
    return zip(*(islice(it, i, None) for i, it in enumerate(tee(sequence, n))))


def format_classifier_outputs(
        predictions: np.array,
        tokens: List[List[str]]
) -> Generator[List[Tuple[str, int]], None, None]:
    """Reformat the classifier outputs.

    Args:
        predictions: The classifier predictions, an array of shape
            `(batch_size, sequence_length, number_of_classes)`.
        tokens: `batch_size` lists of classifier tokens represented as strings.

    Returns:
        A generator that yields lists of `token - label` pairs
        (one for each example in the batch).

    Raises:
        `AssertionError` if the batch size is inconsistent between the `predictions`
        and the `tokens`.
    """
    assert predictions.shape[0] == len(tokens), \
        f"Inconsistent batch sizes of predictions and tokens: {predictions.shape[0]} != {len(tokens)}"
    predicted_labels = np.argmax(predictions, axis=-1)
    for sequence_tokens, sequence_predictions in zip(tokens, predicted_labels):
        yield list(zip(sequence_tokens, sequence_predictions))


def restore_words_and_labels(
        token_data: Iterable[Tuple[str, int]],
        label_map: Mapping[int, str],
        subword_prefix: str = "##"
) -> List[AnnotatedToken]:
    """Create `AnnotatedToken` instances from raw BERT tokens and NER label IDs.

    Args:
         token_data: `(integer, string)` tuples where the string is a subword token form
            and the integer is a NER label ID.
        label_map: A mapping from NER label IDs to the corresponding NER labels.
        subword_prefix: The prefix that distinguishes subword tokens in non-starting positions
            from subword tokens in starting positions. Defaults to `'##'`.

    Returns:
        A list of annotated tokens where words are restored from the subword
        tokens and NER labels are assigned to them. Each word gets the label
        that was assigned to its first subword token.
    """
    words, labels = [], []
    space, hyphen = " ", "-"
    prefix_len = len(subword_prefix)
    for form, label_id in token_data:
        if form == hyphen:
            words.append(form)
        elif len(words) != 0 and words[-1] == hyphen:
            if form.lower() in FUNC_WORDS.conjunctions:
                words.extend([space, form])
                labels.append(label_map[label_id])
            else:
                words.append(form)
        elif not form.startswith(subword_prefix):
            words.extend([space, form])
            labels.append(label_map[label_id])
        else:
            words.append(form[prefix_len:])
    words = "".join(words).lstrip(space)
    return [AnnotatedToken(form=word_form, ner=ner_label)
            for word_form, ner_label in zip(words.split(space), labels)]


def add_tags(sentence: MutableSequence[AnnotatedToken], analyzer: PurePOS) -> None:
    """Add morphological tags to the annotation of a sentence using a `PurePOS` model."""
    word_forms = [token.form for token in sentence]
    tags = (ana[-1] for ana in analyzer.tag_sentence(word_forms))
    for token, tag in zip(sentence, tags):
        token.tag = tag


def get_morph_suffix(
        word: AnnotatedToken,
        analyzer: EmMorphPy,
        tag_pattern: re.Pattern = re.compile(r"\[[A-z0-9/]+]"),
        tag_stops: Sequence[str] = ("[/", "[_")
) -> str:
    """Get the inflexion ending of a word.

    Args:
        word: An annotated word (or token).
        analyzer: An `EmMorph` analyzer instance.
        tag_pattern: A regexp pattern that can be used to identify morphological tags.
        tag_stops: Substrings that the tags of suffix morphemes do not contain.
            Defaults to `('[/', '[_')`, where the first substring indicates a stem and
            the second substring indicates a derivation morpheme.

    Returns:
        The morphological ending as a single string.
    """
    analyses = analyzer.analyze(word.form)
    if len(analyses) == 0:
        return ""
    tag_variations = ["".join(tag_pattern.findall(analysis)) for analysis in analyses]
    fitting_tag_vars = enumerate(
        tag_var for tag_var in tag_variations if word.tag.endswith(tag_var))
    best_tag_idx, _ = max(fitting_tag_vars, key=lambda x: len(x[-1]), default=(0, ""))
    best_analysis = analyses[best_tag_idx]
    # `'='` and `'+'` are special characters used by `EmMorph`
    ana_morphs = (tuple(ana_morph.split("=")) for ana_morph in best_analysis.split("+"))
    suffixes = []
    for ana, morph in ana_morphs:
        if all(tag_stop not in ana for tag_stop in tag_stops):
            suffixes.append(morph.lower())
    suffixes = "".join(suffixes)
    if len(suffixes) != 0 and not suffixes.startswith("-"):
        suffixes = "-" + suffixes
    return suffixes


def restore_spaces(annotated_tokens: MutableSequence[AnnotatedToken]) -> None:
    """Adjust spaces to punctuations. This function modifies the input in place!

    Args:
        annotated_tokens: A sequence of annotated tokens where the `wsafter`
            field will be set.
    """
    bigrams = get_ngrams(annotated_tokens)
    for left, right in bigrams:
        if right.form in RIGHT_PUNCT or left.form in LEFT_PUNCT:
            left.wsafter = ""
    annotated_tokens[-1].wsafter = ""


def select_named_entity(
        target_length: int,
        entity_list: Sequence[Sequence[str]],
        ner_tag: str,
        ner_inner_prefix: str = "I-",
        ner_start_prefix: str = "B-",
        suffix: Optional[str] = None
) -> List[AnnotatedToken]:
    """A helper function to select a named entity from a list
    such that its length is as close as possible to the specified
    length. Return it as a list of annotated tokens.

    Args:
        target_length: The optimal length that the selected named entity should have.
        entity_list: A sequence of named entities where each named entity is a
            sequence of token forms.
        ner_tag: The NER tag (without a prefix) that will be assigned to the output tokens
            (e.g. `'ORG'`).
        ner_inner_prefix: The prefix of the NER label that indicates an inner token of a named entity.
            Defaults to `'I-'`.
        ner_start_prefix: The prefix the the NER label that indicates the first token of a named entity.
            Defaults to `'B-'`.
        suffix: A suffix that will be added to the form of the last word of the selected named entity.
            Optional.

    Returns:
        A list of annotated tokens that make up the selected named entity.
    """
    idx_len_diffs = [(i, abs(target_length - len(entity))) for i, entity in enumerate(entity_list)]
    _, min_diff = min(idx_len_diffs, key=lambda x: x[1])
    selected_idx = rand_choice([idx for idx, len_diff in idx_len_diffs if len_diff == min_diff])
    selected_entity = iter(entity_list[selected_idx])
    del idx_len_diffs
    entity_tokens = [AnnotatedToken(next(selected_entity), ner=ner_start_prefix + ner_tag)]
    inner_tag = ner_inner_prefix + ner_tag
    for token in selected_entity:
        entity_tokens.append(AnnotatedToken(token, ner=inner_tag))
    if suffix is not None and len(entity_tokens) != 0:
        entity_tokens[-1].form += suffix
    return entity_tokens


def replace_named_entities(
        annotated_tokens: Sequence[AnnotatedToken], *,
        replacement_names: Mapping[str, Sequence[Sequence[str]]],
        definite_names: Optional[Mapping[str, Sequence[Sequence[str]]]] = None,
        ner_inner_prefix: str = "I-",
        ner_start_prefix: str = "B-",
        analyzer: Optional[EmMorphPy] = None
) -> List[AnnotatedToken]:
    """Replace the named entities in the input sequence.

    Args:
        annotated_tokens: The input tokens that form a sequence.
        replacement_names: A mapping that maps NER categories to lists from which a name can be
            randomly chosen. These names are represented as sequences of tokens. The named entities
            in the input will be replaced by such names.
        definite_names: A mapping that serves the same purpose as `replacement_names`,
            but elements will be chosen from it only when a named entity is preceded by a definite
            article. If unspecified, only `replacement_names` will be used. Optional.
        ner_inner_prefix: The prefix of the NER label that indicates an inner token of a named entity.
            Defaults to `'I-'`.
        ner_start_prefix: The prefix the the NER label that indicates the first token of a named entity.
            Defaults to `'B-'`.
        analyzer: An `EmMorph` analyzer model that allows to add the morphological suffixes of the
            original named entity to entity with which it is replaced. It is recommended only if the
            entities in `replacement_names` are abstract mask labels. Optional.

    Returns:
        The tokens of the output sequence with the named entities replaced.
    """
    new_tokens = []
    len_named_entity = 0
    ner_prefixes = (ner_start_prefix, ner_inner_prefix)
    has_article = False
    bigrams = get_ngrams(chain(annotated_tokens, (AnnotatedToken(""),)))
    # A dummy token was added above so that the last token can also be a left element of a bigram.
    for left, right in bigrams:
        if left.ner.startswith(ner_prefixes):
            len_named_entity += 1
            if not right.ner.startswith(ner_inner_prefix):
                prefix_len = len(ner_start_prefix) if left.ner.startswith(ner_start_prefix) \
                    else len(ner_inner_prefix)
                tag = left.ner[prefix_len:]
                suffix = get_morph_suffix(left, analyzer) if analyzer is not None else None
                replacement_map = definite_names if definite_names is not None and has_article \
                    else replacement_names
                new_tokens.extend(select_named_entity(
                    target_length=len_named_entity,
                    entity_list=replacement_map[tag],
                    ner_tag=tag,
                    suffix=suffix
                ))
                len_named_entity = 0
                has_article = False
        else:
            new_tokens.append(left)
            if left.form.lower() in FUNC_WORDS.def_articles and right.ner.startswith(ner_start_prefix):
                has_article = True
    return new_tokens


def fix_articles(
        annotated_tokens: MutableSequence[AnnotatedToken],
        ner_inner_prefix: str = "I-",
        ner_start_prefix: str = "B-"
) -> None:
    """Check and fix definite articles in Hungarian.
    If the token `'a'` precedes a named entity that starts with a vowel, replace the token with `'az'`.
    If the token `'az'` precedes a named entity that starts with a consonant, replace the token with `'a'`.
    The letter case of the article will not be changed.
    """
    ner_prefixes = (ner_start_prefix, ner_inner_prefix)
    for left, right in get_ngrams(annotated_tokens):
        left_lowered = left.form.lower()
        if all((
                left_lowered in FUNC_WORDS.def_articles,
                not left.ner.startswith(ner_prefixes),
                right.ner.startswith(ner_start_prefix)
        )):
            new_form = None
            if right.form[0] in VOWELS:
                if left_lowered == FUNC_WORDS.pre_cons_def_article:
                    new_form = FUNC_WORDS.pre_vowel_def_article
            elif left_lowered == FUNC_WORDS.pre_vowel_def_article:
                new_form = FUNC_WORDS.pre_cons_def_article

            if new_form is not None:
                if left.form.islower():
                    left.form = new_form
                elif left.form.istitle():
                    left.form = new_form.title()
                else:
                    left.form = new_form.upper()


def create_byte_ids(
        annotated_tokens: Iterable[AnnotatedToken],
        offset: int,
        sep_id: int,
        eos_id: int,
        no_ner_prefix: str = "O",
        ner_start_prefix: str = "B"
) -> array:
    """Create decoder input IDs from annotated tokens.

    Args:
        annotated_tokens: The input tokens that form a sequence.
        offset: A value by which byte values will be shifted to make room for special tokens.
        sep_id: The ID of the special token that surrounds named entities.
        eos_id: The ID of the `<eos>` (sequence end) token.
        no_ner_prefix: The prefix of the NER label that indicates tokens that are not named entities.
            Defaults to `'O'`.
        ner_start_prefix: The prefix the the NER label that indicates the first token of a named entity.
            Defaults to `'B'`.

    Returns:
        The decoder input IDs.
    """
    input_ids = array("H")
    encoding = "utf-8"
    is_entity_started = False
    wsbefore = []
    for annotated_token in annotated_tokens:
        if annotated_token.ner == no_ner_prefix:
            if is_entity_started:
                input_ids.append(sep_id)
                is_entity_started = False
            input_ids.extend(wsbefore)
        elif annotated_token.ner.startswith(ner_start_prefix):
            if is_entity_started:
                input_ids.append(sep_id)
            else:
                is_entity_started = True
            input_ids.extend(wsbefore)
            input_ids.append(sep_id)
        else:
            input_ids.extend(wsbefore)
        input_ids.extend(byte_val + offset for byte_val in annotated_token.form.encode(encoding))
        wsbefore = (byte_val + offset for byte_val in annotated_token.wsafter.encode(encoding))
    if is_entity_started:
        input_ids.append(sep_id)
    input_ids.append(eos_id)
    return input_ids


def split_id_sequence(
        byte_id_sequence: array,
        eos_id: int
) -> Tuple[array, array]:
    """Split the byte IDs in the condition from the byte IDs in the generated sequence.

    Args:
        byte_id_sequence: The condition and the output generated by a decoder model as
            a single byte ID sequence.
        eos_id: The ID of the `<eos>` (sequence end) token.

    Returns:
        The condition and the generated sequence without `<eos>` token IDs.

    Raises:
        `AssertionError` if the input byte sequence does not contain
        at least 1 `<eos>` token ID.
    """
    assert eos_id in byte_id_sequence, f"Ill-formed input: No <eos> token " \
                                       f"(ID {eos_id}) in the sequence."
    eos_ids = [i for i, byte_id in enumerate(byte_id_sequence) if byte_id == eos_id]
    if len(eos_ids) == 1:
        eos_ids.append(len(byte_id_sequence))
    first_eos, second_eos, *_ = eos_ids
    condition_seq = byte_id_sequence[:first_eos]
    generated_seq = byte_id_sequence[first_eos+1:second_eos]
    return condition_seq, generated_seq


def format_decoder_output(
        condition: array,
        generated: array,
        sep_id: int
) -> array:
    """Reformat the decoder output.

    Args:
        condition: The condition sequence that the decoder model continued.
        generated: The sequence that was generated given the condition.
        sep_id: The ID of the special token that surrounds named entities.

    Returns: A sequence where subsequences in the condition surrounded by
        `<sep>` token IDs are replaced with subsequences from the generated sequence.
        The `<sep>` token IDs are removed in the output.

    Raises:
        `AssertionError` if the number of `<sep>` token IDs in the condition is not
        equal to the number of `<sep>` token IDs in the generated sequence.
    """
    condition_sep_indices = array("I", (i for i, val in enumerate(condition) if val == sep_id))
    if len(condition_sep_indices) == 0:  # If the condition holds, the generated tokens are not needed.
        return array("H", (val for val in condition if val != sep_id))

    generated_sep_indices = array("I", (i for i, val in enumerate(generated) if val == sep_id))
    assert len(condition_sep_indices) == len(generated_sep_indices), \
        f"The number of <sep> tokens in the condition is not equal to the number of <sep> tokens " \
        f"in the generated tokens: {len(condition_sep_indices)} != {len(generated_sep_indices)}."

    condition_sep_indices.insert(0, 0)
    condition_sep_indices.append(len(condition))
    generated_sep_indices.extend((0, 0))
    condition_intervals = zip(*(islice(it, i, None, 2) for i, it in enumerate(tee(condition_sep_indices, 2))))
    generated_intervals = zip(*(islice(it, i, None, 2) for i, it in enumerate(tee(generated_sep_indices, 2))))
    output_tokens = array("H")
    for condition_interval, generated_interval in zip(condition_intervals, generated_intervals):
        # noinspection PyTupleAssignmentBalance
        cond_left, cond_right = condition_interval
        # noinspection PyTupleAssignmentBalance
        gen_left, gen_right = generated_interval
        if cond_left != 0:
            cond_left += 1
        gen_left += 1
        output_tokens.extend(condition[cond_left:cond_right])
        output_tokens.extend(generated[gen_left:gen_right])
    return output_tokens


def decode_output(
        byte_ids: array,
        offset: int,
) -> str:
    """Decode the reformatted output.

    Args:
        byte_ids: A sequence of byte IDs.
        offset: A value by which byte values will be shifted.

    Returns:
        The decoded sequence.
    """
    byte_values = (token_id - offset for token_id in byte_ids)
    byte_values = b"".join(byte_value.to_bytes(1, byteorder="big") for byte_value in byte_values)
    return byte_values.decode("utf-8")
