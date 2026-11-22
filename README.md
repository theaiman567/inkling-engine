# InkEngine v3.0

**A standalone C++17 / CUDA / AVX2 3-tier (VRAM → RAM → SSD) serving engine built from scratch for [Inkling-2-Mini](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (`general.architecture = inkling2`), with zero `llama.cpp` or `ggml` dependencies.**

InkEngine is engineered specifically for consumer GPUs that **cannot fit the full model + long context in VRAM** (e.g., 6–12 GB NVIDIA GPUs + 16 GB RAM + NVMe/SATA SSDs). By exploiting Inkling-2's **3× Looped-MoE architecture** (where 12 physical Core layers execute 3× per token while 12 Outer layers execute 1×), pairing a native **learned MoE expert predictor**, and paging global KV cache across VRAM → RAM → NVMe SSD with GPU Radix-Select Quest attention, InkEngine delivers **3× faster decode**, **2.2× faster prefill**, and **up to 512K–1M token context** on an 8 GB GPU — with **100% lossless token parity**.

---

## Should You Use InkEngine? (Honest Hardware Guide)

Before installing, check where InkEngine shines — and where you are better off using `llama.cpp`, Ollama, or vLLM.

| Your Setup | Should You Use InkEngine? | Why |
|---|---|---|
| **6 GB – 12 GB VRAM GPU**<br>*(RTX 3060 8/12G, 3070, 3080, 4060, 4070, Laptop GPUs)* | **YES — Primary Sweet Spot** | Standard engines statically offload layers across PCIe/CPU and OOM past ~32K–48K context (~8–11 tok/s). InkEngine keeps ~93–94% of active experts in VRAM dynamically, runs 0 SSD expert reads at 128K, and hits **27–35 tok/s** with up to **512K+ context**. |
| **16 GB VRAM GPU**<br>*(RTX 4060 Ti 16G, 4070 Ti Super, 4080, 5070 Ti, 5080)* | **YES for higher quants, 256K–512K+ ctx, or OpenCode** | `UD-IQ3_XXS` (~10.9 GiB) already fits in 16 GB at short context, where `llama.cpp -ngl 99` is already fast. However, InkEngine still gives big wins if you run **higher-precision quants (`Q4_K_M`, `Q5_K_M`, `Q6_K`)** that exceed 16 GB, push **256K–1M context**, or want **zero-VRAM speculative decoding + `<0.2s` persistent NVMe OpenCode prefix caching + 97 captured CUDA Graphs**. |
| **24 GB – 32 GB+ VRAM GPU**<br>*(RTX 3090, 4090, 5090, A6000)* | **Usually NO** *(unless using specific features)* | If your quant + context already fits 100% in VRAM, `llama.cpp` / `ik_llama.cpp` (`-ngl 99`) is simpler and its cuBLAS/MMQ kernels will typically match or beat InkEngine on raw prompt prefill. InkEngine is still useful here only if you want **512K–1M SSD KV context**, **zero-VRAM speculative coding boost (`K ≤ 6`)**, or **persistent disk KV prefix caching across restarts**. |
| **Multi-user / high-concurrency API** | **NO** | InkEngine is a **single-user local engine** (one active generation stream at a time). Use **vLLM** or **SGLang** for batched multi-user production serving. |
| **Models other than Inkling-2** | **NO** | Built exclusively for `general.architecture = inkling2` (Looped-MoE, `AttnRes`, `ShortConv`, `SiTU`, SWA+Global hybrid attention). It will not load Llama, Qwen, DeepSeek, or Gemma GGUFs. |
| **Linux / macOS / AMD / Intel / Apple Silicon** | **Not yet** | Currently **Windows x64 + NVIDIA CUDA (`sm_75+`) + AVX2 CPU** only. The OS-specific surface is small (Win32 `OVERLAPPED` I/O, `VirtualAlloc`, Winsock2) — PRs for Linux POSIX/`io_uring` are welcome. |

### Even When a Model Fits in VRAM, What Unique Speedups Does InkEngine Have?
1. **97 Captured CUDA Graphs + Super-Fused Inkling-2 Kernels**: Fuses `AttnRes` mixing, `LoopControl` projections, `ShortConv`, multi-projection Attention GEMV, and exact FP32 Router+Top-K — eliminating ~624 CUDA kernel launches per token across Inkling-2's 48 logical layers.
2. **Zero-VRAM Lossless Speculative Decoding**: Built-in O(1) rolling 3-gram suffix-hash drafter (`K ≤ 6` tokens/step, always on, 0% slowdown on prose) plus opt-in 1-Loop (24L) GPU-only self-drafting (`--self-draft-k 2|3`) that reuses the 24 captured CUDA Graphs with zero extra VRAM.
3. **Persistent NVMe OpenCode Prefix Caching**: Saves 3-tier system/tool KV checkpoints to disk so restarting the server or rebooting restores `8,000+` agent prompt tokens in **`<0.2s`** instead of re-prefilling for 1–3 minutes.

---

## System Requirements

- **OS**: Windows 10 / 11 (x64)
- **GPU**: NVIDIA RTX GPU with Tensor Cores (**Compute Capability `sm_75` or higher** — RTX 20 / 30 / 40 / 50 series)
- **VRAM**: 6 GB minimum, **8 GB+ recommended**
- **System RAM**: **16 GB minimum** (32 GB+ recommended for larger quants or >256K context)
- **CPU**: x86-64 with **AVX2 + FMA** (used for fast fallback on VRAM-missed experts and `.moepred` predictor inference)
- **Storage**: Any SSD for the GGUF model (even SATA works with 0 steady-state SSD reads at 128K!); **NVMe SSD recommended** for OS `%TEMP%` / `%LOCALAPPDATA%` if using 256K–1M SSD KV paging or persistent KV checkpoints
- **Build Tools**:
  - [NVIDIA CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) (`nvcc` in `PATH` and `CUDA_PATH` set)
  - [Visual Studio 2019 or 2022](https://visualstudio.microsoft.com/downloads/) (Community or Build Tools) with the **Desktop development with C++** workload

### Supported GGUF Tensor Quantization Formats
`IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ4_NL`, `NVFP4`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `F16`, `BF16`, `F32` (covers Unsloth Dynamic `UD-IQ3_XXS` and standard K-quant Inkling-2-Mini GGUFs).

---

## Installation & Build

### 1. Clone the Repository
```cmd
git clone https://github.com/<your-username>/inkling-engine.git
cd inkling-engine
```

### 2. Build for Your NVIDIA GPU Architecture
Run `build.bat` and pass your GPU's CUDA compute capability (`sm_XX`):

| GPU Series | Build Command |
|---|---|
| RTX 20-series (Turing: 2060–2080 Ti) | `build.bat sm_75` |
| RTX 30-series (Ampere: 3060–3090 Ti) | `build.bat sm_86` |
| RTX 40-series (Ada Lovelace: 4060–4090) | `build.bat sm_89` *(default)* |
| RTX 50-series (Blackwell: 5060–5090) | `build.bat sm_120` |

```cmd
build.bat sm_89
```
This compiles `src/cuda_kernels.cu` + the 9 C++17/AVX2 modules into `build\inkengine.exe` (no external libraries beyond `cudart.lib` and `ws2_32.lib`).

### 3. Download the Model GGUF
Download an Inkling-2-Mini GGUF (for 8 GB GPUs, **`Inkling-2-Mini-UD-IQ3_XXS.gguf`** ~10.86 GiB is recommended):
- Hugging Face: [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)

### 4. Download the Learned MoE Expert Predictor *(Optional, Strongly Recommended for ≤12 GB GPUs)*
InkEngine includes a pure C++17/AVX2 runtime for [Lanilen/Inkling-2-Mini-Expert-Predictor](https://huggingface.co/Lanilen/Inkling-2-Mini-Expert-Predictor), which lifts 8 GB VRAM hit rates from ~79% to **~93–94%** (+25% faster decode).

Download these two files from Hugging Face and place them inside `models\`:
```text
inkling-engine\
  models\
    inkling2-pred-v3.moepred
    inkling2-start-v3.moepred
```
InkEngine auto-detects `models\*.moepred` relative to `build\inkengine.exe`. If omitted (or if you pass `--no-predictor`), InkEngine cleanly falls back to its built-in Loop-Aware LFU heuristic.

---

## How to Run

### Option A: OpenAI-Compatible Local Server (Recommended)
Works with **OpenCode**, **Cline**, **Continue**, **Open WebUI**, Python `openai` SDK, or `curl`:

```cmd
build\inkengine.exe --model "C:\path\to\Inkling-2-Mini-UD-IQ3_XXS.gguf" --serve --host 127.0.0.1 --port 8787 --ctx 131072 --effort max
```

Endpoints exposed:
- `POST /v1/chat/completions` (SSE streaming, tool calls, `reasoning_content`, automatic KV prefix caching)
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health` (readiness, `max_ctx`, active `effort`)
- `GET /stats` (live VRAM / RAM / SSD telemetry & expert cache hit rates)
- `POST /shutdown` (clean shutdown)

### Option B: Standalone CLI Prompt
```cmd
build\inkengine.exe --model "C:\path\to\Inkling-2-Mini-UD-IQ3_XXS.gguf" --ctx 65536 --effort medium --prompt "Explain how paged attention works." --max-tokens 512
```

### Option C: One-Command OpenCode CLI (`inkcode`)
`inkcode.ps1` runs InkEngine headlessly in the background (so no browser/GUI eats RAM) and launches the [OpenCode](https://opencode.ai) terminal UI pre-connected to `Inkling-2-Mini`.

1. **Set your model path** (User environment variable so you never have to retype it):
   ```powershell
   [Environment]::SetEnvironmentVariable("INKENGINE_MODEL", "C:\path\to\Inkling-2-Mini-UD-IQ3_XXS.gguf", "User")
   ```
   *(Or pass `--gguf "C:\path\to\model.gguf"` per command.)*

2. **Add InkEngine to `%USERPROFILE%\.config\opencode\opencode.jsonc`**:
   ```jsonc
   {
     "provider": {
       "inkengine": {
         "name": "InkEngine Local",
         "npm": "@ai-sdk/openai-compatible",
         "options": {
           "baseURL": "http://127.0.0.1:8787/v1",
           "apiKey": "inkengine-local"
         },
         "models": {
           "Inkling-2-Mini": {
             "name": "Inkling-2-Mini (Local 3-Tier)",
             "reasoning": true,
             "tool_call": true,
             "limit": { "context": 131072, "output": 16384 }
           }
         }
       }
     }
   }
   ```

3. **Run directly** (or drop a tiny `inkcode.cmd` wrapper in any folder on your `PATH`):
   ```powershell
   .\inkcode.ps1                        # Auto-start server (128K ctx, effort=max) & open OpenCode CLI
   .\inkcode.ps1 --effort low           # Faster replies with lighter thinking (none|minimal|low|medium|high|xhigh|max)
   .\inkcode.ps1 --ctx 65536            # 64K 100%-VRAM exact context
   .\inkcode.ps1 --ctx 262144           # 256K 3-tier VRAM + RAM + NVMe SSD context
   .\inkcode.ps1 --ctx 524288           # 512K 3-tier VRAM + RAM + NVMe SSD context
   .\inkcode.ps1 start                  # Preload background server only
   .\inkcode.ps1 status                 # Live VRAM / RAM / SSD telemetry
   .\inkcode.ps1 stop                   # Shut down server & free all VRAM/RAM
   ```

---

## Tuning for Different Systems

InkEngine automatically queries your free VRAM and system RAM at startup, so most users only need `--model`, `--ctx`, and `--effort`. For custom setups:

| Flag | Default | When to Adjust |
|---|---|---|
| `--ctx <N>` | `65536` (`131072` via `inkcode`) | Set `65536` for 100%-VRAM exact KV on 8 GB GPUs, `131072` for daily coding, or `262144` / `524288` for huge repos via NVMe SSD KV paging. |
| `--effort <lvl>` | `max` (`--serve`) / `none` (`--prompt`) | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` (`0.99`). Use `low` or `medium` for faster responses; `max` for hardest reasoning tasks. |
| `--kv-quant <q4\|q8\|f16>` | `q4` | `q4` maximizes VRAM left for hot experts; `q8` or `f16` uses more KV memory for higher KV precision (great on 12–16 GB+ GPUs). |
| `--vram-kv-mb <N>` | `0` (auto, capped at 640 MiB) | On **12–16 GB+ GPUs**, increase (e.g., `--vram-kv-mb 1280` or `2048`) to keep full 128K–256K KV resident in VRAM. |
| `--ram-kv-mb <N>` | `1024` | On **32 GB+ RAM** systems, raise to `2048`–`4096` to avoid SSD KV writes even at 256K–512K context. |
| `--ram-mb <N>` | `0` (auto, up to ~5.8 GiB) | On **32 GB+ RAM** systems with larger quants (`Q4_K_M`–`Q6_K`), raise (e.g., `--ram-mb 12288`) to keep 100% of cold experts in RAM. |
| `--kv-swap-dir <dir>` | `%TEMP%` | Point to your fastest NVMe drive if `%TEMP%` is on a slower SATA/HDD drive. |
| `--threads <N>` | `10` | Set to your CPU's physical performance-core count (or ~75–85% of logical threads) for fastest CPU fallback on missed experts. |
| `--self-draft-k <0\|2\|3>` | `0` (off) | Opt-in 1-Loop (24L) GPU-only lossless self-speculative drafting. Boosts code/refactoring speed (+7–10%), though cold prose may see lower acceptance. |
| `--no-disk-kv-cache` | *(enabled)* | Disable persistent NVMe OpenCode KV prefix checkpoints in `%LOCALAPPDATA%\InkEngine\kv_cache\`. |
| `--sync-preload` | *(async)* | Wait for full Tier-1 RAM expert preload to finish before opening the HTTP port. |

---

## Verified Benchmarks

Measured on **RTX 4060 8 GB + Ryzen 5 5500 (6C/12T) + 16 GB DDR4 + SATA model SSD + NVMe OS SSD**, running `Inkling-2-Mini-UD-IQ3_XXS.gguf` (10.86 GiB) with `Q4_0` KV cache:

| Benchmark | `llama.cpp` (Unsloth Studio) | InkEngine v1.0 | InkEngine v2.0 | **InkEngine v3.0** | v3.0 vs. `llama.cpp` |
|---|---:|---:|---:|---:|---:|
| **64K steady decode** | 10.64 tok/s *(48K max)* | 26.03 tok/s | 32.34 tok/s | **32.65 tok/s** | **3.07×** |
| **128K full context (`pos=130K`)** | OOM | 20.15 tok/s | 27.57 tok/s | **27.58 tok/s** | **2.59×** *(2.7× ctx)* |
| **256K NVMe SSD (`pos=260K`)** | OOM | 18.82 tok/s | 18.82 tok/s | **26.75 tok/s** | **2.51×** *(5.3× ctx)* |
| **512K NVMe SSD (`pos=520K`)** | OOM | 5.73 tok/s | 5.73 tok/s | **24.43 tok/s** | **2.30×** *(10.8× ctx)* |
| **Code refactoring (`MetricsCalculator`)** | 10.64 tok/s | 26.28 tok/s | 33.81 tok/s | **35.18 tok/s** *(37.87 w/ `--self-draft-k 3`)* | **3.31×–3.56×** |
| **Batched prefill (2K code prompt)** | ~55 tok/s | ~24 tok/s | 71.8 tok/s | **120.69 tok/s** | **~2.19×** |
| **Cold server boot (`128K`)** | ~40s | ~34s | 34.4s | **18.6s** *(async RAM warm)* | **~2.1× faster** |
| **OpenCode Turn 1 (10.5K prompt)** | ~200s | ~440s | 163.3s | **99.6s** *(cold)* / **29.1s** *(NVMe KV hit)* | **~6.9× faster** |

### Exactness & Verification Commands
You can verify token-for-token exactness and predictor math on your own machine:
```cmd
build\inkengine.exe --model "C:\path\to\model.gguf" --ctx 4096 --kv-quant f16 --no-fp16-loop --verify
build\inkengine.exe --model "C:\path\to\model.gguf" --verify-predictor
build\inkengine.exe --model "C:\path\to\model.gguf" --ctx 65536 --bench-prefill 2048
build\inkengine.exe --model "C:\path\to\model.gguf" --ctx 65536 --bench 128
```
- **Model Token Parity**: 100% exact match (`29/29` tokenizer match, `"17 * 23 is 391."`).
- **Predictor Parity**: `pred-v3.vec` = `1.639e-07`, `start-v3.vec` = `2.980e-07` max abs diff vs. PyTorch golden vectors.

---

## Architecture Highlights (Under the Hood)

1. **Loop-Aware 3-Tier Expert Cache + Learned Predictor**: Exploits Inkling-2's 3× core loop (`blk.6..17` executed 36 times/token vs. 12 outer passes). Uses 2-bucket variable-size VRAM/RAM pools (`3.05 MiB` compact + `3.36 MiB` universal) and loop-phased zero-race async DMA swaps during core passes, guided by `pred-v3` + `start-v3`.
2. **Async Background RAM Preload**: Opens the HTTP server in ~18.6s while streaming cold experts from SSD into Tier-1 RAM in the background.
3. **Super-Fused CUDA Kernels + 97 Captured CUDA Graphs**: Single-launch multi-projection Attention GEMV, 3-way LoopControl + RMSNorm, gated `Wo` + `attn_post_conv` residual fusion, and exact FP32 Router+Top-K.
4. **Layer-Wise PCIe-Streamed WMMA Tensor Core Prefill (`B = 128`)**: Cooperative 8-element dequantizer across 10 quant formats feeding `nvcuda::wmma` FP16→FP32 Tensor Cores, with a double-buffered 28 MiB PCIe staging ring for cold outer experts.
5. **3-Tier Hybrid Paged KV (VRAM → RAM → NVMe SSD) + GPU Radix-Select Quest**: Keeps all 40 SWA layers (~45 MiB) pinned in VRAM and pages the 8 Global layers across VRAM, RAM, and Queue-Depth-16 `FILE_FLAG_OVERLAPPED` NVMe SSD using INT8 Quest summaries and O(N) GPU Radix-Select Top-K.
6. **Zero-VRAM Lossless Speculative Decoding**: O(1) rolling 3-gram suffix hash index across the full context (`K ≤ 6`) plus opt-in 1-Loop (24L) GPU-only self-drafting (`--self-draft-k`).
7. **Persistent NVMe OpenCode KV Checkpoints**: Atomic 3-tier prefix checkpoints (`tools-prefix`, pre-`<env>` `static-prefix`, `full-system`) restored in `<200 ms` across restarts.

---

## Credits & Acknowledgments

- **Inkling-2-Mini** model architecture and GGUF quantizations by the Inkling / Unsloth communities.
- **MoE Expert Predictor (`pred-v3` / `start-v3`)** weights and format by [Lanilen/Inkling-2-Mini-Expert-Predictor](https://huggingface.co/Lanilen/Inkling-2-Mini-Expert-Predictor).
- Contributions welcome — especially **Linux POSIX / `io_uring` porting**, additional quant types, and multi-GPU / Q8 KV benchmarks!
