#!/bin/bash

DETECTOR_URL="https://nc.nlp.nytud.hu/s/b6N4DwXGko3sqSR/download"
CORRECTOR_URL="https://nc.nlp.nytud.hu/s/3j87n4rfbRJPx6F/download"

if [[ $# -lt 1 ]]; then
  echo "Please specify the directory where you want to download the models."
  exit 1
fi

MODEL_DIR=$1
if [[ ! -d $MODEL_DIR ]]; then
  mkdir "$MODEL_DIR"
fi

DETECTOR="$MODEL_DIR"/detector
if [[ ! -d "$DETECTOR" ]]; then
  wget -O "$DETECTOR".zip $DETECTOR_URL
  unzip "$DETECTOR".zip -d "$MODEL_DIR"
  rm "$DETECTOR".zip
fi

CORRECTOR="$MODEL_DIR"/corrector
if [[ ! -d "$CORRECTOR" ]]; then
  wget -O "$CORRECTOR".zip $CORRECTOR_URL
  unzip "$CORRECTOR".zip -d "$MODEL_DIR"
  rm "$CORRECTOR".zip
fi
