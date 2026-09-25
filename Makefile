# Optional convenience wrapper around scripts/. NOTE: `make` is not installed on
# the local Fedora box -- use the scripts directly there (./scripts/check-local.sh).
# This Makefile is here for Colab and Kaggle, which do have make.
#
# `check` runs inside a container and needs no GPU.
# `hello` / `vecadd` build AND run, so they need a real GPU.

.PHONY: check hello vecadd all clean venv

# --- local, no GPU needed ---
check:
	@./scripts/check-local.sh

# --- native, needs a real GPU (Colab/Kaggle) ---
all: hello vecadd

hello:
	@./scripts/build.sh src/01_hello/hello.cu
	@./build/hello

vecadd:
	@./scripts/build.sh src/02_vector_add/vector_add.cu
	@./build/vector_add

venv:
	@./scripts/setup-venv.sh

clean:
	rm -rf build
