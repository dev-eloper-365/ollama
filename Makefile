.PHONY: build up down logs restart test clean

# Build the Docker images
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d

# Start with development overrides
up-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Stop all services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Restart services
restart:
	docker-compose restart

# Pull Ollama models (run after services are up)
pull-models:
	docker-compose exec ollama ollama pull embeddinggemma
	docker-compose exec ollama ollama pull gemma3:12b

# Test the API
test:
	curl http://localhost:8080/health
	@echo "\n✅ Testing embeddings endpoint..."
	curl -X POST http://localhost:8080/api/embed \
		-H "Content-Type: application/json" \
		-d '{"texts": ["Hello world"]}'

# Clean up volumes and containers
clean:
	docker-compose down -v
	docker system prune -f

# Show service status
status:
	docker-compose ps

