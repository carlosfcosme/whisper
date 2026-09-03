# Offline, no-download CI/dev entry points.
# Helper services bind localhost only (whisper/localhost.py).

export WHISPER_OFFLINE ?= 1
export HF_HUB_OFFLINE ?= 1
export TRANSFORMERS_OFFLINE ?= 1
export HF_DATASETS_OFFLINE ?= 1
export WHISPER_BIND_HOST ?= 127.0.0.1

PYTEST ?= pytest
TOX ?= tox
PYTEST_OFFLINE_ARGS ?= --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda and not requires_weights'

.PHONY: help test-offline tox-offline ci-offline

help:
	@echo "test-offline   Run no-download offline pytest (localhost binds)"
	@echo "tox-offline    Run the tox 'offline' env"
	@echo "ci-offline     make test-offline && make tox-offline"

test-offline:
	$(PYTEST) $(PYTEST_OFFLINE_ARGS)

tox-offline:
	$(TOX) -e offline

ci-offline: test-offline tox-offline
