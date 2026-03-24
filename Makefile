# TT-Forge Compiletron - Makefile for common operations

.PHONY: help build run test clean shell detect models stats quick compile-quick compile-parallel

# Default target
help:
	@echo "TT-Forge Compiletron - Make targets"
	@echo "===================================="
	@echo ""
	@echo "Docker Operations:"
	@echo "  build           Build Docker image"
	@echo "  run             Run with docker-compose"
	@echo "  stop            Stop docker-compose"
	@echo "  clean           Remove containers and volumes"
	@echo ""
	@echo "Testing & Info:"
	@echo "  test            Run test suite in container"
	@echo "  detect          Detect hardware in container"
	@echo "  shell           Interactive shell in container"
	@echo ""
	@echo "Model Operations:"
	@echo "  models          List all models"
	@echo "  stats           Show model statistics"
	@echo "  quick           Show quick test models"
	@echo ""
	@echo "Compilation:"
	@echo "  compile-quick   Compile 5 fastest models"
	@echo "  compile-parallel  Compile in parallel on all chips"
	@echo ""
	@echo "Local Operations (no Docker):"
	@echo "  test-local      Run tests locally"
	@echo "  detect-local    Detect hardware locally"

# Docker operations
build:
	@./docker-build.sh

run:
	docker-compose up -d

stop:
	docker-compose down

clean:
	docker-compose down -v
	docker rmi tt-forge-compiletron:latest || true

# Testing
test:
	@./docker-run.sh test

test-local:
	@./run_tests.sh

# Hardware detection
detect:
	@./docker-run.sh detect

detect-local:
	@python3 compiletron.py detect

# Interactive shell
shell:
	@./docker-run.sh shell

# Model operations
models:
	@./docker-run.sh models list

stats:
	@./docker-run.sh models stats

quick:
	@./docker-run.sh models quick

# Compilation
compile-quick:
	@./docker-run.sh compile --quick

compile-parallel:
	@./docker-run.sh compile --parallel --count 50

# Docker image info
info:
	@docker images | grep tt-forge-compiletron || echo "Image not built yet"
	@echo ""
	@docker volume ls | grep compiletron || echo "No volumes yet"
