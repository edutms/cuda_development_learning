# CUDA development on a free GPU

Learning CUDA C++ without owning an NVIDIA GPU: write and compile-check locally,
run kernels on a free cloud GPU.

```
   edit .cu locally  ->  check-local.sh  ->  git push  ->  Colab runtime runs on a T4
   (VS Code)            (podman, no GPU)   (GitHub)      (free GPU, driven from VS Code)
```

The split matters. `nvcc` needs a GPU to *run* code but not to *compile* it, so the
compile loop — which is where almost all of the early iterations go — happens locally
in seconds. Colab's limited free GPU hours get spent only on actually executing kernels.

## One-time setup

**1. Local Python env** (for authoring notebooks; nothing GPU-related):

```bash
./scripts/setup-venv.sh
```

**2. VS Code + Colab extensions** — this is what lets you drive a free T4 without
leaving the editor. See **[docs/vscode-colab.md](docs/vscode-colab.md)** for the walkthrough.

```bash
code --install-extension ms-toolsai.jupyter
code --install-extension Google.colab
```

**3. GitHub remote** — the Colab runtime pulls your code from GitHub. Already configured
here (`git remote -v` → `edutms/cuda_development_learning`); a fresh clone needs nothing.
The repo must be **public** for the notebook's clone cell to work without auth.

## Daily loop

```bash
code src/03_whatever/kernel.cu      # write
./scripts/check-local.sh            # compile-check locally: ~4s, no GPU needed
git add -A && git commit -m "..." && git push
```

Then in `notebooks/00_colab_setup.ipynb` (open in VS Code, kernel = Colab T4): re-run the
clone/pull cell, then the build/run cells.

**Why the git step does not go away:** the Colab runtime is a different machine and cannot
read your working tree. Pushing is how code gets there — see the gotchas below.

The browser remains a fallback:
[open in Colab](https://colab.research.google.com/github/edutms/cuda_development_learning/blob/main/notebooks/00_colab_setup.ipynb)
→ **Runtime → Change runtime type → T4 GPU**.

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
| `docs/vscode-colab.md` | Driving a Colab T4 from VS Code — setup and limits. |
| `docs/setup-plan.md` | Why this is built the way it is — design rationale and decisions. |

`./scripts/check-local.sh` compiles everything and is the command you run constantly.
A `Makefile` is included too (`make check`, `make hello`, `make vecadd`), but `make` is
not installed on this machine — it is there for convenience in Colab and Kaggle, where it
is. The `hello` / `vecadd` targets build *and run*, so they need a real GPU either way.

## Free GPU options

| | GPU | Arch flag | Quota | Local IDE? |
|---|---|---|---|---|
| **Google Colab** | T4 (16 GB) | `sm_75` | Best-effort, ~15–30 h/week, ~12 h sessions | **Yes — official extension** |
| **Lightning AI Studios** | T4 and up | detected | ~80 credit-h/mo ≈ **22 h on a T4** | Yes — SSH, plus a real terminal |
| **Kaggle Notebooks** | T4 or P100 | `sm_75` / `sm_60` | Guaranteed 30 h/week, 9 h sessions | No |

Colab's free GPU is *not guaranteed* — at busy times you may be offered CPU only.
Kaggle is the fallback and needs no changes: `scripts/build.sh` detects the architecture
at build time, so the same repo compiles correctly on a P100.

Compute capability → flag: T4 `sm_75`, P100 `sm_60`, A100 `sm_80`, L4 `sm_89`.

### Lightning AI Studios — when you need a persistent machine

Not needed for the everyday loop: the Colab extension already gives you a T4 inside VS Code.
Reach for a Studio when you need what Colab runtimes structurally cannot provide — a
**persistent filesystem, a real terminal, and sanctioned SSH**. A Studio survives restarts
and GPU switches, and installs (`apt-get`, pip, conda, source builds) persist, so a toolchain
is a one-time cost.

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

Their rendered docs are JavaScript-only stubs — **append `.md` to any docs URL** to get
readable markdown. Start at
[modify-environment](https://lightning.ai/docs/platform/build/ai-studio/modify-environment)
(persistence rules) and
[on-start-actions](https://lightning.ai/docs/platform/build/ai-studio/on-start-actions)
(`on_start.sh` vs `.studiorc` — env vars set in the former do *not* reach later terminals,
which is the trap).

One caveat before committing time: it is unverified whether a Studio ships `nvcc` at all.
"No CUDA setup" normally means driver plus CUDA *runtime* for PyTorch, and the *compiler*
is a separate package such images often omit. Check `nvcc --version` first.

## Gotchas worth knowing

**The Colab kernel cannot see your local files.** This is the one thing that surprises
people using the VS Code extension: "a local notebook path is not the same thing as a
remote runtime path." The runtime is a different machine, so it cannot read your working
tree — which is exactly why cell 2 of the notebook (`git clone` / `git pull`) is mandatory
rather than a convenience. Edit locally, push, pull in the runtime.

**SSH tunnels into Colab are still prohibited.** An official extension is not the same
thing as sanctioned SSH. `colab-ssh`, `remocolab` and `colabcode` remain disallowed by the
[Colab FAQ](https://research.google.com/colaboratory/faq.html) on free runtimes — "remote
control through SSH shells or remote desktops" and "bypassing the notebook interface" can
be terminated without warning, and Google actively breaks these tools. If you need a real
terminal on a GPU box, use Lightning AI Studios, not a tunnel.

**Colab's "Connect to local runtime" is the opposite of what it sounds like.** It runs the
kernel on *your* machine with Colab as the frontend — giving you no GPU at all here. Not
to be confused with the VS Code extension, which is the direction you want.

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
