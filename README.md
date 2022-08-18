# anonymizer_hu
The Hungarian anonymization tool for CURLICAT

## Description

The anonymization tool can handle named entities in 2 different ways:

1. Stem masking: Each detected named entity is masked with a tag corresponding to the NER category of the named entity (e. g. PER, LOC, ORG, MISC).
However, the morphological suffixes of the last word of the named entity are also identified and concatenated with the NER tag.
2. Replacement: Each named entity is replaced with another named entity selected from a predefined list (see `./models/entity_map.json` and `./models/definite_entity_map.json`). The new named entity comes from the same NER category as the original. The new named entity is automatically put in the grammatical form required by its context.

The _replacement_ method is implemented by using a neural decoder model (GPT-2) and it may be unstable. The current solution is automatically falling back to the _stem masking_ method should any problem occur.

When run in a docker container (see the [next section](#docker)), the anonymization method can be selected by specifying the `USE_CORRECTOR` environment variable (set it to `true` to select the _replacement_ method).

The input must be plain text, processing CONLL documents is not supported yet.


## Docker

### Build

Run the following commands to build the docker image:

```bash
git clone https://github.com/nytud/anonymizer_hu.git
cd anonymizer_hu
make build_docker
```

If you do not want the model files to be downloaded each time a new container is started, you can run `./docker/install.sh ./models` before the build command.

### Start a container

Use `make run_docker` or write a `docker run` command, e. g.

```bash
docker run --rm -d --name anonymizer_hu --gpus '"device=0"' -e USE_CORRECTOR=true -p <port>:5000 nytud/anonymizer:latest
```

The `docker_run` command above specifies the environment variable `USE_CORRECTOR` that determines which anonymization method will be applied to the input.

Specifying a GPU makes sense even if you do not use the neural decoder required for the _replacement_ anonymization method as the detector (the model that identifies named entities) is a neural network as well.

### Request

The application reacts to POST API requests, for example:

```bash
$ curl -X 'POST' '<port>:5000/anonymize' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Some text", "format": "text"}
```

where `<port>` is the port on which the application is running. A free port is identified automatically if the container is started with `make run_docker`.

The application returns a JSON response:

```json
{
  "original_text": "Some text",
  "anonymized_text": "Same text but anonymized",
  "format": "text"
}
```

### Stop the container

Use either `make stop_docker` or `docker container stop anonymizer_hu`.

