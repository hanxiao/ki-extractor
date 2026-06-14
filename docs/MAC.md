# Run on Apple Silicon (Mac, Metal)

The Knowledge Graph Extractor runs natively on an Apple Silicon Mac with **no Docker and no NVIDIA
GPU**. Homebrew's `llama.cpp` `llama-server` serves the model on **Metal**, and the FastAPI app +
the `jina-embeddings-v5-text-nano` dedup embedder run in a local `uv` virtualenv. No application
logic changes are needed: the model is decoupled behind the OpenAI-compatible `LLAMA_URL`, so the
only Mac-specific concerns are which GGUF to use, the `llama-server` flags, and installing the deps
the Docker image normally bundles.

Tested on an **M3 Pro / 36 GB**, macOS, `llama.cpp` build 9430 (Homebrew). At least **32 GB** of
unified memory is recommended - the Q3_K_XL model wires ~17 GB.

## What's different from the NVIDIA/Docker path

| Area | NVIDIA / Docker | Apple Silicon | Why |
| --- | --- | --- | --- |
| Serving | `llama.cpp:server-cuda` container | Homebrew `llama-server` (Metal) | No CUDA / `nvidia-container-toolkit` on macOS. |
| GPU offload | implicit (CUDA) | `-ngl 999` (`NGL` env) | Unified memory: put all layers on Metal. |
| Flash attention | `--flash-attn 1` | `--flash-attn on` | This build's flag takes `on\|off\|auto`, not `1`. |
| Speculative decode | `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1` | Same (`SPEC_ARGS`) | MTP works on `llama.cpp` >= 9430. Leave `SPEC_ARGS` empty to disable. |
| App + embedder | Docker container | `python app.py` in a `uv` venv; embedder on CPU | No GPU passthrough into Docker on macOS; CPU keeps Metal free for the LLM. |
| torch | from the CPU wheel in the image | `uv pip install torch` (MPS/CPU build) | Installed into the local venv instead. |

## Prerequisites

```bash
brew install llama.cpp          # Metal build of llama-server
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv (Python env)
uv --version
```

You also need a free **Jina API key** (https://jina.ai/api-key) and, recommended, a free
**Hugging Face read token** (https://huggingface.co/settings/tokens) for a fast, stable model
download.

## 1. Install the Python deps

The Dockerfile's dependency set, into a local venv. `torch` pulls the Apple-Silicon (MPS/CPU) build.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  fastapi uvicorn httpx numpy python-multipart pypdf \
  sentence-transformers peft einops \
  torch huggingface-hub
```

## 2. Download the model (~17 GB)

```bash
mkdir -p models
# MTP GGUF (recommended - enables draft-mtp speculative decoding with llama.cpp >= 9430)
HF_TOKEN=hf_your_token \
.venv/bin/python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('unsloth/Qwen3.6-35B-A3B-MTP-GGUF','Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf',local_dir='models')"
```

The `HF_TOKEN=` prefix is optional but avoids the unauthenticated rate limit. If your `llama.cpp`
is older than build 9430 (check `llama-server --help` for `draft-mtp`), use the **non-MTP** GGUF
(`unsloth/Qwen3.6-35B-A3B-GGUF`) instead and leave `SPEC_ARGS` empty.

## 3. Set your key

```bash
cp .env.example .env
sed -i '' 's/^JINA_API_KEY=.*/JINA_API_KEY=jina_your_real_key/' .env
```

Mac defaults are baked into `scripts/mac-run.sh`; override any in `.env`:

```bash
MODEL_FILE=Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
CTX_SIZE=16384            # matches the Docker default; raise if you have headroom
SPEC_ARGS='--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1'
LLAMA_URL=http://127.0.0.1:8080
JOBS_DIR=./data/jobs
```

## 4. Run

```bash
bash scripts/mac-run.sh
```

Starts `llama-server` (Metal) on `:8080`, waits for it to load, then starts the app on `:3000`.
Open **http://localhost:3000/**.

Stop the app with `Ctrl+C`; stop the model with `pkill -f llama-server`.

## Memory notes

On a 36 GB machine the Q3_K_XL model wires ~17 GB. It runs comfortably; if a long job pages
heavily, lower `CTX_SIZE` in `.env`. The embedder stays on CPU precisely so Metal's memory is
reserved for the LLM.

| Unified memory | Qwen3.6-35B-A3B (Q3, ~17 GB) | Guidance |
| --- | --- | --- |
| 16 GB | ✗ won't fit | Not supported; a 7-14B model is the ceiling on this tier. |
| 24 GB | ⚠ tight | Fits but keep `CTX_SIZE` modest; close Chrome/Docker/IDEs. |
| 32-48 GB | ✓ comfortable (**tested on 36 GB**) | Recommended. |
| 96 GB+ | ✓ plenty | Room for much longer context or larger models. |

## Alternative backend: MLX (faster prefill)

`scripts/mac-run.sh` can serve `:8080` with Apple's **mlx-lm** instead of llama.cpp, behind a
`BACKEND` knob. It's opt-in; the default stays llama.cpp.

```bash
BACKEND=mlx bash scripts/mac-run.sh      # or set BACKEND=mlx in .env
```

On an M3 Pro / 36 GB serving the 4-bit MLX model, prefill is dramatically faster (~6x vs llama.cpp)
with decode at or above the MTP path. Extraction is prefill-heavy (each round re-reads the full
document), so this is the win that matters for large docs and zips.

| Backend | Engine | Model | Prefill | Context ceiling |
| --- | --- | --- | --- | --- |
| `llamacpp` *(default)* | llama.cpp (Metal, GGUF) | `models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | baseline | high (stable) |
| `mlx` | mlx-lm (`mlx_lm.server`) | `models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit` | ~6x | ~75-85K (auto-capped) |

### MLX setup (one-time)

mlx-lm must live in its **own** venv - installing it into the app `.venv` bumps `transformers` and
can break the embedder.

```bash
uv venv .venv-mlx
VIRTUAL_ENV=$PWD/.venv-mlx uv pip install mlx-lm
# Download or convert a 4-bit MLX build into models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit
# (e.g. mlx_lm.convert, or pull a pre-quantized 4-bit MLX repo).
```

`mac-run.sh` checks both the venv and the model exist before starting and errors with the fix if not.

### MLX env knobs

| Var | Default | Meaning |
| --- | --- | --- |
| `BACKEND` | `llamacpp` | `llamacpp` or `mlx`. |
| `MLX_MODEL` | `models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit` | Path to the MLX model dir. |
| `MLX_CTX_CAP` | `75000` (fp16 KV) / `85000` (4-bit KV) | Max context for MLX; `CTX_SIZE` is auto-capped to this. |
| `MODEL_NAME` | (auto = `MLX_MODEL` under `mlx`) | Pinned so the app's request `model` field matches the loaded model. |

### MLX caveats

- **Context auto-cap.** Stock `mlx_lm.server` runs an **fp16 KV cache** with no KV-quantization
  flag (see upstream [ml-explore/mlx-lm#1043](https://github.com/ml-explore/mlx-lm/issues/1043)).
  As of mlx-lm 0.31.3 (latest release) there is still **no `--kv-bits` on `mlx_lm.server`**, so
  this is the path everyone gets today: fp16 KV OOMs at ~78-92K actual tokens on 36 GB, and the
  script caps context to `MLX_CTX_CAP` (75K) accordingly. The script is forward-looking - it
  auto-detects a `--kv-bits` flag and, when present, switches to 4-bit KV and raises the cap
  toward ~85K. That flag isn't in any released mlx-lm yet; I'm working on landing it upstream, and
  until it ships the fp16/75K path is the one in use.
- **`MODEL_NAME` is pinned to the model path.** `mlx_lm.server` resolves the request's `model`
  field against the loaded model and otherwise tries to fetch it from HuggingFace (a request for
  the friendly label `qwen3.6` -> 404). llama.cpp ignores the label; MLX needs the match, so the
  script exports `MODEL_NAME=$MLX_MODEL` for this backend (default unchanged for llama.cpp).

Stop the MLX model with `pkill -f mlx_lm.server`.
