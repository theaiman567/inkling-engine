# InkEngine v4.1

**A standalone C++17 / CUDA / AVX2 3-tier (VRAM → RAM → SSD) multimodal serving engine built from scratch for [Inkling-2-Mini](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (`general.architecture = inkling2` + `inkling2v` vision), with zero `llama.cpp` or `ggml` dependencies.**

InkEngine is engineered specifically for consumer GPUs that **cannot fit the full model + long context in VRAM** (e.g., 6–12 GB NVIDIA GPUs + 16 GB RAM + NVMe/SATA SSDs). By exploiting Inkling-2's **3× Looped-MoE architecture** (where 12 physical Core layers execute 3× per token while 12 Outer layers execute 1×), pairing a native **learned MoE expert predictor**, single-pass Q8 activation quantization + `G=8`/`G=4` sub-warp MoE kernels + zero-copy spin-poll dispatch, paging global KV cache across VRAM → RAM → NVMe SSD with O(1) free-list clock eviction, VRAM-pinned multi-chunk exact attention and GPU Radix-Select Quest attention, and streaming **multimodal vision (`mmproj-F16.gguf`)** on-demand through GPU Tensor Cores with **0 MiB permanent VRAM/RAM overhead**, InkEngine delivers **3.6× faster decode (`38 tok/s` at 64K, `31 tok/s` at 512K SSD!)**, **2.2× faster prefill**, **native image understanding**, and **up to 512K–1M token context** on an 8 GB GPU — backed by an automated **8-gate long-context & fault-injection verification suite** and **hardened local HTTP Bearer authentication**.

**New in v4.1 ("Agent Flow")**: built for long OpenCode agent sessions. Every generated turn is replayed **token-for-token**, so the next request reuses its KV cache instead of re-processing it (a `write` follow-up reused **980/991** prompt tokens, where v4.0 re-prefilled the turn). File writes and other tool calls now **stream their arguments live** (265 deltas instead of one block after minutes of silence). The model's previous thinking is kept even when a client sends it as `reasoning` or `thinking`, and every request logs decode cost, speculative gain, expert hit rate and the exact cause of any cache miss. See `CHANGELOG.md`.

---

## Should You Use InkEngine? (Honest Hardware Guide)

Before installing, check where InkEngine shines — and where you are better off using `llama.cpp`, Ollama, or vLLM.

| Your Setup | Should You Use InkEngine? | Why |
|---|---|---|
| **6 GB – 12 GB VRAM GPU**<br>*(RTX 3060 8/12G, 3070, 3080, 4060, 4070, Laptop GPUs)* | **YES — Primary Sweet Spot** | Standard engines statically offload layers across PCIe/CPU and OOM past ~32K–48K context (~8–11 tok/s). InkEngine keeps ~93–94% of active experts in VRAM dynamically, runs 0 SSD expert reads at 128K, and hits **31–41 tok/s** with up to **512K+ context**. |
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
git clone https://github.com/theaiman567/inkling-engine.git
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
This compiles `src/cuda_kernels.cu` + the 10 C++17/AVX2 modules into `build\inkengine.exe` (no external libraries beyond standard CUDA & Windows system libs: `cudart.lib`, `ws2_32.lib`, `windowscodecs.lib`, `ole32.lib`).

### 3. Download the Model GGUF *(+ Optional Vision Projector)*
Download an Inkling-2-Mini GGUF (for 8 GB GPUs, **`Inkling-2-Mini-UD-IQ3_XXS.gguf`** ~10.86 GiB is recommended):
- Hugging Face: [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
- **Multimodal Vision *(Optional, Recommended)***: Also download **`mmproj-F16.gguf`** (~610 MB) from the same repo and place it in the **same folder** as your model GGUF. InkEngine auto-detects it at startup with **0 MiB permanent VRAM and 0 MiB permanent RAM overhead** — weights stream layer-by-layer on demand through the engine's existing 14 MiB PCIe staging buffer onto GPU Tensor Cores, and all temporary activation VRAM is freed immediately after encoding each image.

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

Endpoints exposed *(protected by default with `Authorization: Bearer <token>`, Host allowlist, and non-wildcard CORS; pass `--api-key <token>` or `--no-auth` to override)*:
- `POST /v1/chat/completions` (SSE streaming with UTF-8 boundary buffering, **multimodal `image_url` base64 PNG/JPEG/BMP/WebP**, tool calls, `reasoning_content`, strict HTTP 400 `context_length_exceeded` guard, automatic KV prefix caching — including multi-turn image KV reuse!)
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health` (readiness, `max_ctx`, active `effort`, `vision` status; responds in `<1 ms` even under stalled sockets)
- `GET /stats` (live VRAM / RAM / SSD telemetry & expert cache hit rates)
- `POST /shutdown` (authenticated clean shutdown)

### Option B: Standalone CLI Prompt *(Text or Image)*
```cmd
build\inkengine.exe --model "C:\path\to\Inkling-2-Mini-UD-IQ3_XXS.gguf" --ctx 65536 --effort medium --prompt "Explain how paged attention works." --max-tokens 512
build\inkengine.exe --model "C:\path\to\Inkling-2-Mini-UD-IQ3_XXS.gguf" --ctx 16384 --effort low --image "C:\path\to\screenshot.png" --prompt "Inspect this UI screenshot for layout bugs."
```

### Option C: One-Command OpenCode CLI (`inkcode`)
`inkcode.ps1` runs InkEngine headlessly in the background (so no browser/GUI eats RAM) and launches the [OpenCode](https://opencode.ai) terminal UI pre-connected to `Inkling-2-Mini`.

1. **Set your model path** (User environment variable so you never have to retype it):
   ```powershell
   [Environment]::SetEnvironmentVariable("INKENGINE_MODEL", "C:\path\to\Inkling-2-Mini-UD-IQ3_XXS.gguf", "User")
   ```
   *(Or pass `--gguf "C:\path\to\model.gguf"` per command.)*

2. **Add InkEngine to `%USERPROFILE%\.config\opencode\opencode.jsonc`** *(on startup, `inkcode.ps1` automatically generates a per-user CSPRNG token in `%LOCALAPPDATA%\InkEngine\api_token` and syncs only `provider.inkengine.options.apiKey`)*:
   ```jsonc
   {
     "provider": {
       "inkengine": {
         "name": "InkEngine Local",
         "npm": "@ai-sdk/openai-compatible",
         "options": {
           "baseURL": "http://127.0.0.1:8787/v1",
           "apiKey": "inkengine-local",
           "timeout": false,
           "headerTimeout": false,
           "chunkTimeout": false
         },
         "models": {
           "Inkling-2-Mini": {
             "name": "Inkling-2-Mini (Local 3-Tier)",
             "reasoning": true,
             "tool_call": true,
             "modalities": { "input": ["text", "image"], "output": ["text"] },
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
| `--mmproj <path>` | *(auto-detected)* | Explicit path to `mmproj-F16.gguf` if stored outside your model GGUF directory (or set `INKENGINE_MMPROJ` for `inkcode`). |
| `--image <path>` | *(none)* | Attach one or more local PNG/JPEG/BMP/WebP images to standalone `--prompt` CLI calls *(local paths are intentionally disabled over HTTP `image_url` for security)*. |
| `--no-vision` | *(vision ON)* | Disable multimodal vision auto-detection even if `mmproj-F16.gguf` is present. |
| `--api-key <token>` | *(random CSPRNG)* | Set an explicit Bearer token for `--serve` (`inkcode.ps1` auto-manages `%LOCALAPPDATA%\InkEngine\api_token` and syncs OpenCode). |
| `--no-auth` | *(auth ON)* | Disable Bearer token enforcement on `--serve` (only for trusted isolated environments). |
| `--cors-origin <url>` | *(CORS blocked)* | Allow a single browser origin (e.g., `http://localhost:3000`); default blocks cross-origin requests (`403`) and DNS rebinding. |
| `--verify-suite` | *(off)* | Run the automated 8-gate parity & robustness suite (SWA wrap, batched prefill, speculative, Turn-Rewind/Disk v3, R4 multi-chunk exact KV, R6 fault-injected I/O, R8 Quest rollback, R10 predictor bookkeeping, R14 GPU sampling). |
| `--sync-preload` | *(async)* | Wait for full Tier-1 RAM expert preload to finish before opening the HTTP port. |

---

## Verified Benchmarks

Measured on **RTX 4060 8 GB + Ryzen 5 5500 (6C/12T) + 16 GB DDR4 + SATA model SSD + NVMe OS SSD**, running `Inkling-2-Mini-UD-IQ3_XXS.gguf` (10.86 GiB) with `Q4_0` KV cache:

| Benchmark | `llama.cpp` (Unsloth Studio) | InkEngine v1.0 | InkEngine v3.2 (Pre-Audit) | **InkEngine v4.0** | v4.0 vs. `llama.cpp` |
|---|---:|---:|---:|---:|---:|
| **64K steady decode** | 10.64 tok/s *(48K max)* | 26.03 tok/s | 32.65 tok/s | **38.03 tok/s** *(+16.5% vs v3.2)* | **3.57×** |
| **128K full context (`pos=130K`)** | OOM | 20.15 tok/s | 27.58 tok/s | **31.84 tok/s** *(+15.4% vs v3.2)* | **2.99×** *(2.7× ctx)* |
| **256K NVMe SSD (`pos=260K`)** | OOM | 18.82 tok/s | 26.75 tok/s | **30.85 tok/s** *(+15.3% vs v3.2)* | **2.90×** *(5.3× ctx)* |
| **512K NVMe SSD (`pos=520K`)** | OOM | 5.73 tok/s | 24.43 tok/s | **30.98 tok/s** *(+26.8% vs v3.2!)* | **2.91×** *(10.8× ctx)* |
| **Code editing (speculative R13+R16)** | 10.64 tok/s | 26.28 tok/s | 35.18 tok/s | **41.35 tok/s** *(73.8% acc / +25% vs scalar)* | **3.89×** |
| **Batched prefill (2K code prompt)** | ~55 tok/s | ~24 tok/s | 120.69 tok/s | **122.22 tok/s** *(R15 direct pinned DMA)* | **~2.22×** |
| **Cold server boot (`128K`)** | ~40s | ~34s | 18.6s | **18.0s–21.7s** *(async RAM warm + DMA pin)* | **~2.0× faster** |
| **OpenCode Turn 1 (10.5K prompt)** | ~200s | ~440s | 99.6s / 29.1s | **~98s** *(cold)* / **~28s** *(NVMe KV hit)* | **~7.1× faster** |
| **Sampling (`T=1.0, min_p=0.0`)** | Host scan | ~312 µs | ~312 µs | **~46 µs** *(bit-exact GPU pre-filter)* | **6.8× faster sampling** |
| **Multimodal Vision (`mmproj-F16`, idle)** | ~610+ MiB VRAM/RAM | N/A | 0 MiB VRAM/RAM | **0 MiB permanent VRAM / 0 MiB RAM** | **Zero text slowdown** |
| **Full HD 1080p encode + multi-turn cache** | Re-encodes or pins VRAM | N/A | ~3.0s / 10 ms | **~3.0s** *(SATA stream)* / **0.0 ms + 10 ms KV hit** | **1,493 VRAM slots kept** |

*(v4.1 changes the serving path, not the kernels, so the raw-speed rows above are the v4.0 measurements. They are synthetic single-token benchmarks: real OpenCode agent turns measured **~25 tok/s** scalar decode in a 107K-token session, and the new per-request telemetry shows speculative decoding currently costs time on reasoning-heavy turns (gain 0.76–0.93×). Cost-aware drafting is the next fix.)*

| Agent-flow check (v4.1, real model, `--ctx 16384`) | v4.0 | **v4.1** |
|---|---:|---:|
| `write` tool-call arguments streamed to client | 1 delta (at the end) | **265 deltas (live)** |
| Next-turn KV reuse after a `write` | miss → turn re-prefilled | **980 / 991 tokens reused** |
| Prior thinking sent back as `reasoning` | dropped (prompt 507 tok) | **kept (prompt 991 tok)** |

*(Note: Official `min_p = 0.0` sampling applies a `1e-6` relative tail floor (`p < 1e-6 × p_max`), dropping negligible tail probability mass while keeping GPU candidate pre-filtering bit-exact and sub-50 µs.)*

### Exactness & Verification Commands
You can verify token-for-token exactness, all 8 long-context / fault-injection / sampling parity gates, and predictor math on your own machine:
```cmd
build\inkengine.exe --model "C:\path\to\model.gguf" --ctx 4096 --kv-quant f16 --no-fp16-loop --verify --verify-suite
build\inkengine.exe --model "C:\path\to\model.gguf" --ctx 4096 --kv-quant q4 --verify-suite
build\inkengine.exe --model "C:\path\to\model.gguf" --verify-predictor
build\inkengine.exe --model "C:\path\to\model.gguf" --ctx 65536 --bench-prefill 2048
build\inkengine.exe --model "C:\path\to\model.gguf" --ctx 65536 --bench 128
```
- **Model Token Parity**: tokenizer `29/29` exact match. The golden greedy answer currently prints `"17 * 23 = 391."` instead of the reference `"17 * 23 is 391."` on both v4.0 and v4.1 (a pre-existing near-tie flip, likely from v4.0's numerically-equivalent kernel changes; tracked separately). The answer is still correct.
- **8-Gate Long-Context & Robustness Suite (`--verify-suite`, `F16` & `Q4_0`)**: SWA ring-wrap `L0_SWA_err=0.000001`, post-wrap `B=128/32/7` Top-1 exact match, Speculative ON vs. OFF `96/96` exact (`77/77 = 100%` past `pos=1024`), **bit-exact (`0.000000`)** Turn-Rewind + Disk v3 Checkpoint restore, **0.000000 error** multi-chunk exact global attention under 6-VRAM-frame / SSD pressure (`R4`), 4/4 fault-injected checked I/O & atomic publish tests (`R6`), bit-exact Quest extrema rollback across 64-token page boundaries (`R8`), exact predictor route replay (`R10`), and **200/200 bit-exact** GPU sampling pre-filter (`R14`).
- **Predictor Parity**: `pred-v3.vec` = `1.639e-07`, `start-v3.vec` = `2.980e-07` max abs diff vs. PyTorch golden vectors.

---

## Architecture Highlights (Under the Hood)

1. **Loop-Aware 3-Tier Expert Cache + Learned Predictor**: Exploits Inkling-2's 3× core loop (`blk.6..17` executed 36 times/token vs. 12 outer passes). Uses 2-bucket variable-size VRAM/RAM pools (`3.05 MiB` compact + `3.36 MiB` universal) and loop-phased zero-race async DMA swaps during core passes, guided by `pred-v3` + `start-v3`.
2. **Async Background RAM Preload**: Opens the HTTP server in ~18.6s while streaming cold experts from SSD into Tier-1 RAM in the background.
3. **Single-Pass Q8 Quantization, Sub-Warp MoE (`G=8`/`G=4`), Zero-Copy Spin-Poll & 97 Captured CUDA Graphs**: Quantizes activation vectors once per step into L2-resident staging with `<1` load/thread smem broadcast, runs 0%-idle-lane `G=8` `gate_up` & 0-smem `G=8`/`G=4` `down` MoE kernels, fuses single-split attention sink/normalization, and replaces per-layer WDDM event waits with zero-copy `MoESplitPacket::seq` spin-polling.
4. **Layer-Wise Direct-Pinned PCIe WMMA Tensor Core Prefill (`B = 128`)**: Cooperative 8-element dequantizer across 10 quant formats feeding `nvcuda::wmma` FP16→FP32 Tensor Cores, streaming cold outer experts directly from `cudaHostRegister`'d Tier-1 RAM → VRAM with zero CPU bounce `memcpy`.
5. **3-Tier Hybrid Paged KV (VRAM → RAM → NVMe SSD) + O(1) Free-List + Multi-Chunk Exact + GPU Radix-Select Quest**: Keeps all 40 SWA layers (`1,152`-slot wrap-safe ring, `50.6 MiB` at `Q4_0`) pinned in VRAM with disjoint per-global-layer page tables, and pages the 8 Global layers across VRAM, RAM, and Queue-Depth-16 `FILE_FLAG_OVERLAPPED` NVMe SSD with O(1) frame free-lists + 64-frame Clock-LRU eviction, per-slot VRAM frame pinning, single-sink online-softmax chunk merging, atomic full-byte I/O publish, exact Quest extrema rollback, and O(N) GPU Radix-Select Top-K.
6. **Zero-VRAM Lossless Adaptive Speculative Decoding & Bit-Exact GPU Sampling**: Adaptive 3/4/5-gram suffix hash index (`K ≤ 6` scaled by acceptance EMA) verified via zero-sync / spin-poll indirect batch MoE (zero `atomicAdd`, `G=4` LM Head) with exact token-order predictor route replay, opt-in 1-Loop (24L) GPU-only self-drafting (`--self-draft-k`), and a bit-exact GPU superset candidate pre-filter (`~46 µs`, `6.8×` faster than `800 KB` host logit copy).
7. **Persistent NVMe OpenCode KV Checkpoints (v3)**: Atomic 3-tier prefix checkpoints (`tools-prefix`, pre-`<env>` `static-prefix`, `full-system`) restored in `<200 ms` across restarts, plus `~55 MiB` zero-Global-copy `~4–10 ms` bit-exact in-place turn-rewind snapshots.
8. **Zero-Overhead GPU-Streamed Multimodal Vision (`inkling2v`)**: Pure Windows WIC decoder + 6 custom CUDA ViT/projector kernels (23-layer Perception-Encoder ViT, exact 2D RoPE, tiled FlashAttention, 2×2 spatial merge, `tok_norm` bypass) streamed in 8–9 MiB slices through the existing prefill PCIe staging ring — leaving VRAM expert slots, RAM budget, and text decode speed 100% untouched. Deterministic hash-derived synthetic token IDs let OpenCode multi-turn conversations reuse both cached image embeddings (`0.0 ms`) and KV prefix caches (`~10 ms`) automatically.
9. **Hardened Local HTTP/SSE Runtime**: Default CSPRNG Bearer auth across all routes, `Host` allowlist & non-wildcard CORS (blocking DNS rebinding), HTTP `image_url` local/UNC file exfiltration guard, strict anti-smuggling header parser, `10s/30s` socket timeouts, bounded 16-worker concurrency so `/health` stays `<1 ms`, UTF-8 multi-byte SSE boundary buffering, and strict `context_length_exceeded` handling.
10. **Agent-Flow Serving (v4.1)**: Token-exact replay of every generated assistant turn (keyed by tool-call ids, spliced on special-token boundaries) so multi-turn agent sessions never re-prefill their own output; official-template tool-call argument rendering (sorted top-level keys, original nested order); live OpenAI `function.arguments` streaming for tool calls; `reasoning_content` / `reasoning` / `thinking` accepted so Inkling always gets its prior thinking back; and per-request telemetry (scalar vs speculative step cost, decode-only expert hit rate, replayed turns, cache-miss diagnostics).

---

## Credits & Acknowledgments

- **Inkling-2-Mini** model architecture and GGUF quantizations by the Inkling / Unsloth communities.
- **MoE Expert Predictor (`pred-v3` / `start-v3`)** weights and format by [Lanilen/Inkling-2-Mini-Expert-Predictor](https://huggingface.co/Lanilen/Inkling-2-Mini-Expert-Predictor).
- Contributions welcome — especially **Linux POSIX / `io_uring` porting**, additional quant types, and multi-GPU / Q8 KV benchmarks!
