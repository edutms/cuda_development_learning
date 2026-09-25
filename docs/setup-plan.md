# Setup plan and design rationale

Reference record of how this environment was designed and built, and *why* it is
shaped the way it is. The [README](../README.md) covers day-to-day usage; this
document covers the reasoning, so that future changes do not undo a decision
without knowing what it was for.

Built 2026-09-25. Marked **as-built** — where implementation diverged from the
original plan, the divergence is recorded in [Deviations](#deviations-from-the-original-plan).

---

## Context

The project started as a single `cuda.ipynb` whose first real cell was `!nvidia-smi`
— and it failed. The machine has an Intel CometLake-U UHD iGPU and no NVIDIA device,
so there is nothing for CUDA to run on locally, and no `nvcc` to compile with.

The goal: study CUDA C++ on a free GPU, without pretending the local machine has one.

## Environment facts (verified at build time)

| | |
|---|---|
| GPU | Intel CometLake-U UHD only — **no NVIDIA device** |
| OS | Fedora 43, **SELinux enforcing** |
| Python | **3.14.7 only** (no 3.12/3.13 fallback) |
| Present | `git`, `podman`, `code`, user-level `jupyter` |
| Absent | `nvcc`, `nvidia-smi`, `make`, **any C compiler**, `gh`, `docker`, `uv`, `pipx` |
| Disk | 179 GB free on `/home` |

## The core decision

**`nvcc` needs a GPU to *run* code, but not to *compile* it.**

That single fact drives the whole design. The compile loop — where nearly all early
iterations go while learning — runs locally in a container in ~4 seconds. Colab's
scarce, best-effort free GPU hours are spent only on actually executing kernels.

```
edit .cu locally  ->  check-local.sh  ->  git push  ->  Colab pulls & runs on a T4
(VS Code)            (podman, no GPU)   (GitHub)      (free GPU)
```

## Decisions and rationale

### CUDA C++ / nvcc, not Numba or CuPy
Chosen deliberately: the goal is to learn the CUDA execution model — memory hierarchy,
occupancy, coalescing — which the Python abstractions hide. CuPy/Numba remain a fine
later addition for quick experiments; they are simply not the study target.

### GitHub as the sync mechanism
Over Google Drive mount (Fedora has no official Drive client, so getting files in is
manual) and over self-contained `%%writefile` notebooks (source trapped in cells, no
real version history). Git gives history, works identically on Kaggle, and makes the
local↔cloud boundary explicit.

The repo must be **public** for the Colab clone cell to work without authentication.

### Explicit `-arch`, detected at build time
The most important implementation detail. Colab sometimes ships a CUDA *toolkit*
newer than its *driver*, and a binary carrying only PTX then fails at load:

```
the provided PTX was compiled with an unsupported toolchain
```

...because the older driver cannot JIT the newer PTX. `scripts/build.sh` queries
`nvidia-smi --query-gpu=compute_cap` and passes `-arch=sm_<real cc>`, which emits
**SASS for the actual device** and skips the JIT path entirely.

This also makes one script correct everywhere: a T4 on Colab (`sm_75`), a P100 on
Kaggle (`sm_60`), an L4 or A100 if offered, and the GPU-less container (falls back
to `sm_75`). Verified with `cuobjdump --dump-sass build/hello` → `arch = sm_75`.

### `nvcc4jupyter` / `%%cuda` magic — rejected
The commonly recommended path, but it has a documented track record of silently
producing no output and of failing to parse `-arch` flags in its argument parser.
Since `-arch` is load-bearing here (previous section), that failure mode is
disqualifying. Plain `nvcc` on real files is more reliable and keeps code in version
control rather than inside notebook cells.

### Podman container over a local CUDA toolkit install
Installing CUDA on Fedora 43 means fighting repo availability for a toolkit that can
never run anything locally. The container is pinned, disposable, and matches Colab's
environment more closely. Two Fedora-specific details are required:

- **`:Z`** on the bind mount — SELinux relabel; without it the container cannot read `src/`.
- **`--entrypoint bash`** — suppresses the image's "NVIDIA Driver was not detected"
  banner. That warning is expected and harmless: compiling needs no driver.

### Local venv scope
JupyterLab only. No GPU libraries — they would not work here. The venv exists purely
so notebooks can be authored and previewed locally; execution is always remote.

## As-built layout

```
cuda_development/
├── README.md                      usage, quotas, gotchas
├── Makefile                       optional wrapper (make absent locally; Colab has it)
├── requirements.txt               jupyterlab only
├── docs/setup-plan.md             this file
├── scripts/
│   ├── build.sh                   arch-detecting nvcc wrapper; CUDA_COMPILE_ONLY=1 to check
│   ├── check-local.sh             podman compile sweep over every .cu
│   └── setup-venv.sh              creates .venv
├── src/
│   ├── common/cuda_check.cuh      CUDA_CHECK, CUDA_CHECK_KERNEL, print_device_info
│   ├── 01_hello/hello.cu          smallest launch — proves the toolchain
│   └── 02_vector_add/vector_add.cu  full host/device cycle + CUDA-event timing
└── notebooks/00_colab_setup.ipynb Colab entry point: clone -> build -> run
```

`scripts/build.sh` is the single build path, shared by the container and Colab.
`CUDA_ARCH` overrides detection; `CUDA_COMPILE_ONLY=1` checks without linking.

## Deviations from the original plan

| Planned | Actual | Consequence |
|---|---|---|
| Container ~3 GB | **7.28 GB** | Longer first pull; cached thereafter |
| `make` as the interface | **`make` is not installed** (nor any C compiler) | Shell scripts became primary; Makefile kept for Colab/Kaggle |
| — | Added `scripts/setup-venv.sh` | Replaced the `make venv` target that could not run |
| — | Added `--entrypoint bash` | Suppresses the misleading driver warning |
| Python 3.14 wheel lag was a flagged risk | **Did not materialize** — JupyterLab 4.6.4 installed cleanly | No fallback to system jupyter needed |

## Verification performed

Confirmed locally:

- Both examples compile for `sm_75`; `vector_add` also for `sm_60` via `CUDA_ARCH`.
- Full sweep takes **~4 s** once the image is cached.
- **Negative test** — injected a missing `;`, got `hello.cu(16): error: expected a ";"`
  and a nonzero exit. Errors genuinely surface rather than passing silently.
- `cuobjdump --dump-sass build/hello` → `arch = sm_75`: real device code, not PTX.
- Rootless podman maps container root to the host user; no file-ownership problems.
- JupyterLab 4.6.4 runs; notebook validates as nbformat 4; `build/` and `.venv/` ignored.

**Not yet verified — requires the GitHub remote:** actual kernel execution on a Colab
T4. The end-to-end proof is `vector_add` printing `PASSED` with a bandwidth figure.

## Remaining steps

1. Create an empty **public** repo `cuda_development` at <https://github.com/new>
   (or `sudo dnf install gh && gh auth login` and use `gh repo create`).
2. `git remote add origin https://github.com/YOUR_USERNAME/cuda_development.git`
   then `git push -u origin main`.
3. Edit `REPO_URL` in cell 2 of `notebooks/00_colab_setup.ipynb`.
4. Open in Colab, set **Runtime → Change runtime type → T4 GPU**, Run all.

## Where this goes next

The examples stop where the environment is proven — they are a toolchain test, not a
curriculum. A natural progression: 2D indexing and matrix multiply → shared-memory
tiling → reductions and warp primitives → streams and overlapping transfers →
profiling with Nsight Compute. *Programming Massively Parallel Processors*
(Hwu, Kirk, Hajj) follows roughly this order and suits this setup.

Note that `build.sh` already passes `-lineinfo`, so `compute-sanitizer` and Nsight
can map results back to source lines whenever profiling starts.
