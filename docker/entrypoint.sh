#!/bin/bash

bash /app/docker/install.sh /app/models
python3 /app/src/anonymizer/api.py
