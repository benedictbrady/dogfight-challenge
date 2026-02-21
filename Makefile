.PHONY: build test run serve dev viz tournament validate analyze sweep clean pyenv train train-eval export dashboard-parse dashboard-export

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

# Start the backend WebSocket server (ORT_DYLIB_PATH enables neural policy)
PORT ?= 3001
ORT_DYLIB_PATH ?= $(shell find .venv -name 'libonnxruntime*.dylib' 2>/dev/null | head -1)
serve:
	ORT_DYLIB_PATH=$(ORT_DYLIB_PATH) cargo run -p dogfight --release -- serve --port $(PORT)

# Install viz dependencies
viz-install:
	cd viz && npm install

# Start the frontend dev server
viz:
	cd viz && npm run dev

# Start both backend and frontend
dev:
	@echo "Starting backend on port $(PORT)..."
	@ORT_DYLIB_PATH=$(ORT_DYLIB_PATH) cargo run -p dogfight --release -- serve --port $(PORT) &
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

# Sweep physics parameters
SWEEP_PARAM ?=
SWEEP_STEPS ?= 11
SWEEP_SEEDS ?= 5
SWEEP_POLICIES ?= chaser,ace,brawler
SWEEP_OUTPUT ?=
sweep:
	cargo run -p dogfight --release -- sweep \
		$(if $(SWEEP_PARAM),--param $(SWEEP_PARAM),) \
		--steps $(SWEEP_STEPS) --seeds $(SWEEP_SEEDS) \
		--policies $(SWEEP_POLICIES) \
		$(if $(SWEEP_OUTPUT),--output $(SWEEP_OUTPUT),)

# Build PyO3 Python module (requires maturin + venv)
pyenv:
	cd crates/pyenv && maturin develop --release

# Train RL agent
CURRICULUM ?= do_nothing:50,dogfighter:100,chaser:150,ace:200
train:
	cd training && python train.py --curriculum $(CURRICULUM)

# Evaluate trained model against all opponents
CKPT ?= training/checkpoints/final.pt
train-eval:
	cd training && python eval.py $(CKPT)

# Export trained model to ONNX
export:
	cd training && python export_onnx.py $(CKPT)

# Parse TensorBoard events into dashboard JSON
RUN_DIR ?= training/runs
dashboard-parse:
	cd training && python parse_tb.py $(if $(RUN),$(RUN),--all ../$(RUN_DIR))

# Export a checkpoint to ONNX for dashboard replay
DASHBOARD_CKPT ?=
DASHBOARD_OUT ?=
dashboard-export:
	cd training && python export_onnx.py $(DASHBOARD_CKPT) -o $(DASHBOARD_OUT) --no-validate

# Clean build artifacts
clean:
	cargo clean
	rm -rf viz/.next viz/out
