#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create an anonymizer API using `fastapi`."""


import uvicorn
from http import HTTPStatus
from typing import Literal
from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from anonymizer.model import setup_model
from anonymizer.constants import (
    SOURCES_ROOT,
    DETECTOR,
    CORRECTOR,
    MASK_MAP,
    ENTITY_MAP,
    DEFINITE_ENTITY_MAP,
    LABEL_MAP,
    TOKENIZER,
)


class Item(BaseModel):
    text: str
    format: Literal["text", "conll"] = "text"

    # @validator("text")
    # def check_text_length(cls, val: str) -> str:
    #     """Check that the text length measured in bytes does not precede the maximal value."""
    #     if len(val.encode("utf-8")) > MAX_INPUT_LENGTH:
    #         raise ValueError(
    #             f"The input text is too long, the maximal length in bytes is {MAX_INPUT_LENGTH}.")
    #     return val


app = FastAPI()
model = setup_model(
    tokenizer_path=TOKENIZER,
    detector_path=DETECTOR,
    corrector_path=CORRECTOR,
    mask_map_path=MASK_MAP,
    label_map_path=LABEL_MAP,
    entity_map_path=ENTITY_MAP,
    definite_entity_map_path=DEFINITE_ENTITY_MAP,
)


@app.post("/anonymize")
def predict(req: Item):
    return JSONResponse(content={
        "original_text": req.text,
        "anonymized_text": model(req.text),
        "format": "text"
    })


@app.get("/", response_model=None, status_code=204)
def test():
    # noinspection PyUnresolvedReferences
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5000,
                reload=True, reload_dirs=[SOURCES_ROOT])
