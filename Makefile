.PHONY: build test run serve dev viz tournament validate analyze clean

# Build the Rust workspace in release mode
build:
	cargo build --release

# Run all tests
test:
	cargo test --workspace

# Run a single match (override P0, P1, SEED as needed)
P0 ?= chaser
P1 ?= dogfighter
SEED ?= 42
run:
	cargo run -p dogfight --release -- run --p0 $(P0) --p1 $(P1) --seed $(SEED)

# Start the backend WebSocket server
PORT ?= 3001
serve:
	cargo run -p dogfight --release -- serve --port $(PORT)

# Install viz dependencies
viz-install:
	cd viz && npm install

# Start the frontend dev server
viz:
	cd viz && npm run dev

# Start both backend and frontend
dev:
	@echo "Starting backend on port $(PORT)..."
	@cargo run -p dogfight --release -- serve --port $(PORT) &
	@sleep 1
	@echo "Starting frontend..."
	@cd viz && npm run dev

# Run a tournament
POLICIES ?= chaser,dogfighter,do_nothing
ROUNDS ?= 10
tournament:
	cargo run -p dogfight --release -- tournament --policies $(POLICIES) --rounds $(ROUNDS)

# Validate an ONNX model
MODEL ?= model.onnx
validate:
	cargo run -p dogfight --release -- validate $(MODEL)

# Analyze battle dynamics
ANALYZE_POLICIES ?= chaser,dogfighter,ace,brawler
ANALYZE_SEEDS ?= 5
LABEL ?=
analyze:
	cargo run -p dogfight --release -- analyze --policies $(ANALYZE_POLICIES) --seeds $(ANALYZE_SEEDS) $(if $(LABEL),--label $(LABEL),) --randomize

# Clean build artifacts
clean:
	cargo clean
	rm -rf viz/.next viz/out
