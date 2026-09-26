# Running Colab GPUs from VS Code

Google shipped an [official Colab extension for VS Code](https://marketplace.visualstudio.com/items?itemName=Google.colab)
([announcement, 2026-09-22](https://developers.googleblog.com/google-colab-is-coming-to-vs-code/)).
It connects a local notebook to a Colab runtime, so you get a free T4 without opening a
browser tab.

This replaces the older advice that VS Code could not reach a Colab runtime. It could not,
until this extension existed.

## Setup

**1. Install both extensions.** The Colab extension depends on the official Jupyter
extension and will prompt for it; installing both up front avoids the prompt.

```bash
code --install-extension ms-toolsai.jupyter
code --install-extension Google.colab
```

**2. Open the notebook.**

```bash
code notebooks/00_colab_setup.ipynb
```

**3. Pick the Colab kernel.** Click **Select Kernel** (top right of the notebook) →
**Colab** → create a new Colab server → choose accelerator **T4** → sign in with your
Google account. Free tier offers T4 and TPU v5e; Pro tiers expose premium accelerators.

**4. Run the cells top to bottom.** `nvidia-smi` should report a Tesla T4, `hello` prints
per-thread lines, and `vector_add` prints `PASSED` with a bandwidth figure.

## The one thing that will confuse you

**The runtime cannot see your local files.** It is a different machine — "a local notebook
path is not the same thing as a remote runtime path." Your editor is local; the kernel is
in Google's cloud.

So you cannot point the Colab kernel at `src/02_vector_add/vector_add.cu` on your laptop.
That is why cell 2 does a `git clone` / `git pull`, and why it is mandatory rather than a
convenience. The loop stays:

```
edit locally  ->  ./scripts/check-local.sh  ->  git push  ->  re-run the pull cell
```

If you want to prove this to yourself once, try reading a local-only path from a cell and
watch it fail. It makes the rest of the design make sense.

## What you gain and what you don't

**Gain:** one window. Real editor for the notebook, VS Code's Python tooling, no browser
tab, and the notebook's outputs sit next to your source tree.

**Don't gain:** a synced filesystem, a terminal on the GPU box, or persistence. Colab
runtimes stay ephemeral, no GPU is guaranteed, and the extension "does not provide a
persistent SSH-style cloud machine." For those, see the Lightning AI Studios section in the
[README](../README.md).

## Still prohibited

An official extension is not sanctioned SSH. `colab-ssh`, `remocolab` and `colabcode`
remain disallowed by the [Colab FAQ](https://research.google.com/colaboratory/faq.html) on
free runtimes — "remote control through SSH shells or remote desktops" and "bypassing the
notebook interface" — and can be terminated without warning.

## Troubleshooting

The extension is new, so expect rough edges. Connection failures ("remote server not
reachable") are a known early class of problem — check the
[issue tracker](https://github.com/googlecolab/colab-vscode/issues) before assuming your
setup is at fault.

If no GPU is offered at all, that is Colab capacity, not your configuration. The browser
notebook and Kaggle are both unaffected fallbacks.
