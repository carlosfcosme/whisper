# Offline clone-to-test target: no WAN, no weight fetch, 127.0.0.1 binds only.
PYTHON ?= python

.PHONY: test-offline test-offline-bootstrap

test-offline test-offline-bootstrap:
	PYTHON=$(PYTHON) bash .github/scripts/test_offline_bootstrap.sh
