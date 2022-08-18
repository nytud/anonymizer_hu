all: usage

usage:
	@echo "Please specify a command: build_docker, run_docker or stop_docker"
.PHONY: usage

# Build the docker image
build_docker:
	docker build . --tag nytud/anonymizer:latest
.PHONY: build_docker

# Start a container
run_docker:
	@make -s stop_docker
	@free_port=$$(./docker/free_port_finder.sh) ; \
		if [ -z "$${free_port}" ] ; then echo "ERROR: no free port found." ; exit 1 ; fi ; \
		docker run --name anonymizer_hu -p "$${free_port}:5000" --rm -d nytud/anonymizer:latest ; \
		echo "OK: anonymizer container running on port $${free_port}" ;
.PHONY: run_docker

## Stop the container
stop_docker:
	@if [ "$$(docker container ls -f name=anonymizer_hu -q)" ] ; then \
		docker container stop anonymizer_hu ; \
	else echo "Nothing to stop." ; \
	fi
.PHONY: stop_docker
