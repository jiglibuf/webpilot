PY ?= .venv/bin/python
TASK ?= Find the cheapest product in the shop and report its name and price.

.PHONY: help install install-browsers test test-all lint clean demo demo-local measure serve-shop probe-input

help:
	@echo "make install          - create .venv and install webpilot (editable) + dev deps"
	@echo "make install-browsers - download the Playwright Chromium build"
	@echo "make test             - offline test-suite (no network, no API keys)"
	@echo "make test-all         - offline + live provider smoke tests (needs API keys)"
	@echo "make demo TASK='...'  - run the agent against a real site with a visible browser"
	@echo "make demo-local       - run the agent against the bundled offline test shop"
	@echo "make serve-shop       - start the bundled test shop on :8765"
	@echo "make measure          - re-run the page-perception measurements into docs/measurements.json"
	@echo "make probe-input      - check whether synthetic input reaches the page in this environment"

install:
	uv venv --python 3.11 .venv
	$(PY) -m ensurepip -q || true
	uv pip install -e ".[dev]"

install-browsers:
	$(PY) -m playwright install chromium

test:
	$(PY) -m pytest -q

test-all:
	$(PY) -m pytest -q -m "live or not live"

measure:
	$(PY) scripts/measure_perception.py

probe-input:
	$(PY) scripts/probe_input_delivery.py

serve-shop:
	$(PY) -m tests.fixtures.site.server --port 8765

demo:
	$(PY) -m webpilot.cli "$(TASK)"

demo-local: TASK = Open the local shop, search for a desk product, add the most expensive result to the cart and report the cart total.
demo-local:
	$(PY) -m tests.fixtures.site.server --port 8765 & echo $$! > /tmp/webpilot-shop.pid
	@sleep 1
	WEBPILOT_PROVIDER=$${WEBPILOT_PROVIDER:-openai} $(PY) -m webpilot.cli "$(TASK)"
	@kill $$(cat /tmp/webpilot-shop.pid) 2>/dev/null || true

clean:
	rm -rf .pytest_cache **/__pycache__ .ruff_cache transcripts screenshots
