FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y curl unzip wget openjdk-11-jdk hfst
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
ENV PATH="${PATH}:/root/.poetry/bin"

COPY . /app
WORKDIR /app

RUN poetry config virtualenvs.create false --local && \
    poetry build && \
    poetry install

RUN chmod +x ./docker/*.sh
ENTRYPOINT ["./docker/entrypoint.sh"]
