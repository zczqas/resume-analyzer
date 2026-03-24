SHELL := /bin/bash

HOST ?= 127.0.0.1
PORT ?= 8000
APP ?= app.main:app

.PHONY: dev down

dev:
	uv run uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

down:
	-pkill -f "uvicorn $(APP) --host $(HOST) --port $(PORT)" || true
	-pkill -f "uvicorn $(APP)" || true
	@echo "Stopped uvicorn (if it was running)."