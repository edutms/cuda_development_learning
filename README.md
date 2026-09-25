# CUDA development on a free GPU

Learning CUDA C++ without owning an NVIDIA GPU: write and compile-check locally,
run kernels on a free cloud GPU.

```
   edit .cu locally  ->  check-local.sh  ->  git push  ->  Colab pulls & runs on a T4
   (VS Code)            (podman, no GPU)   (GitHub)      (free GPU)
```

The split matters. `nvcc` needs a GPU to *run* code but not to *compile* it, so the
compile loop — which is where almost all of the early iterations go — happens locally
in seconds. Colab's limited free GPU hours get spent only on actually executing kernels.

## One-time setup

**1. Local Python env** (for authoring notebooks; nothing GPU-related):

```bash
./scripts/setup-venv.sh
```

**2. GitHub repo** — Colab pulls your code from GitHub, so it needs a remote.
Create an empty **public** repo named `cuda_development` at <https://github.com/new>
(public so the Colab clone cell works without auth), then:

```bash
git remote add origin https://github.com/edutms/cuda_development_learning.git
git add -A && git commit -m "CUDA study environment" && git push -u origin main
```

**3. Open the notebook in Colab** (`REPO_URL` in cell 2 already points here):

```
https://colab.research.google.com/github/edutms/cuda_development_learning/blob/main/notebooks/00_colab_setup.ipynb
```

In Colab: **Runtime → Change runtime type → T4 GPU** before running anything.

## Daily loop

```bash
code src/03_whatever/kernel.cu      # write
./scripts/check-local.sh            # compile-check locally: ~4s, no GPU needed
git add -A && git commit -m "..." && git push
```

Then re-run cell 2 in the Colab notebook to pull, and the build/run cells to execute.

## What's here

| Path | Purpose |
|---|---|
| `scripts/build.sh` | Compiles one `.cu`. Detects the GPU's arch; same script local and remote. |
| `scripts/check-local.sh` | Compile-checks every `.cu` in a podman CUDA container. |
| `scripts/setup-venv.sh` | Creates `.venv` with JupyterLab for local notebook authoring. |
| `src/common/cuda_check.cuh` | `CUDA_CHECK` / `CUDA_CHECK_KERNEL` macros, device info dump. |
| `src/01_hello/` | Smallest kernel launch — proves the toolchain. |
| `src/02_vector_add/` | Full host/device cycle with CUDA-event timing. |
| `notebooks/00_colab_setup.ipynb` | Colab entry point: clone → build → run. |
| `docs/setup-plan.md` | Why this is built the way it is — design rationale and decisions. |

`./scripts/check-local.sh` compiles everything and is the command you run constantly.
A `Makefile` is included too (`make check`, `make hello`, `make vecadd`), but `make` is
not installed on this machine — it is there for convenience in Colab and Kaggle, where it
is. The `hello` / `vecadd` targets build *and run*, so they need a real GPU either way.

## Free GPU options

| | GPU | Arch flag | Quota | Local IDE? |
|---|---|---|---|---|
| **Lightning AI Studios** | T4 and up | detected | ~80 credit-h/mo ≈ **22 h on a T4** | **Yes — VS Code over SSH** |
| **Google Colab** | T4 (16 GB) | `sm_75` | Best-effort, ~15–30 h/week, ~12 h sessions | No |
| **Kaggle Notebooks** | T4 or P100 | `sm_75` / `sm_60` | Guaranteed 30 h/week, 9 h sessions | No |

Colab's free GPU is *not guaranteed* — at busy times you may be offered CPU only.
Kaggle is the fallback and needs no changes: `scripts/build.sh` detects the architecture
at build time, so the same repo compiles correctly on a P100.

Compute capability → flag: T4 `sm_75`, P100 `sm_60`, A100 `sm_80`, L4 `sm_89`.

### Lightning AI Studios — real VS Code on a real GPU

The one free option that lets you stay in your editor. A Studio is a persistent cloud
workspace; you connect **local VS Code to it over SSH** and get a real filesystem,
terminal and `nvcc`. SSH and "connect any IDE" are supported free-tier features here,
not workarounds.

Setup (see [Lightning's connect-local-IDE docs](https://lightning.ai/docs/overview/ai-studio/connect-local-ide)
for the current click-path):

1. Sign up at <https://lightning.ai> and verify your phone — that unlocks the free GPU credits.
2. Create a Studio, then use its SSH option to register your key. You already have one:
   `~/.ssh/id_ed25519.pub` (the same key GitHub uses).
3. In VS Code install **Remote - SSH**, then connect to the host Lightning gives you.
4. In the Studio terminal: `git clone git@github.com:edutms/cuda_development_learning.git`
5. `./scripts/build.sh src/02_vector_add/vector_add.cu && ./build/vector_add`

`build.sh` needs no changes — it detects whatever GPU the Studio is running.

**Make the 22 hours last.** Two habits, both of which this repo is already built around:

- **Switch the Studio to a CPU machine when you are not running kernels.** The filesystem
  persists across machine switches (everything under `~` / `/teamspace/studios/this_studio`),
  so you keep your work and stop burning GPU credits while reading, editing or debugging.
- **Keep compile-checking locally.** `./scripts/check-local.sh` costs nothing and catches
  the compile errors before you ever spend a GPU minute. This stays your first line of
  defence no matter which cloud you run on.

Colab remains the zero-setup fallback, and the notebook still works. Lightning is the
better daily driver; Colab is better for a quick throwaway check.

## Gotchas worth knowing

**You cannot attach VS Code to a Colab runtime.** Google exposes no Jupyter endpoint for
it, and the SSH-tunnel workarounds (`colab-ssh`, `remocolab`, `colabcode`) are explicitly
disallowed by the [Colab FAQ](https://research.google.com/colaboratory/faq.html) on free
runtimes — "remote control through SSH shells or remote desktops" and "bypassing the
notebook interface" can be terminated without warning, and Google actively breaks these
tools. Colab's "Connect to local runtime" is the *opposite* of what it sounds like: it
runs the kernel on your machine, which has no NVIDIA GPU. Use Lightning AI Studios above
if you want an IDE on a GPU.

**`/content` is ephemeral.** Colab wipes the filesystem on disconnect. Git is the only
thing that persists — never leave work only in the runtime.

**Toolkit newer than driver.** Colab sometimes ships a CUDA toolkit ahead of its driver,
and you get:

```
the provided PTX was compiled with an unsupported toolchain
```

This happens when the binary carries only PTX and the driver must JIT it. `build.sh`
passes an explicit `-arch=sm_<real cc>`, which emits SASS for the actual device and
avoids the JIT path entirely. If you hand-roll an `nvcc` command, pass `-arch` yourself.

**`nvcc4jupyter` / `%%cuda` magic.** Widely recommended, but has a track record of
silently producing no output and of failing to parse `-arch` flags. This repo uses plain
`nvcc` on real files instead — more reliable, and it keeps your code in version control
rather than trapped inside notebook cells.

**Local `nvcc` needs the container.** There is no CUDA toolkit — or any C compiler — on
this machine, and no reason to install one; `check-local.sh` runs nvcc inside
`nvidia/cuda:*-devel` (~7 GB, pulled once). The bind mount uses `:Z` for SELinux, which
Fedora requires, and the entrypoint is overridden to `bash` to suppress the container's
"NVIDIA Driver was not detected" banner — that warning is expected and harmless here,
since compiling needs no driver.

## Where to go next

The examples stop at the point where the environment is proven. A natural progression
from here: 2D indexing and matrix multiply → shared-memory tiling → reductions and warp
primitives → streams and overlapping transfers → profiling with Nsight Compute.
*Programming Massively Parallel Processors* (Hwu, Kirk, Hajj) follows roughly this order
and pairs well with this setup.
