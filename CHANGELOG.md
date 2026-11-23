# Changelog / PR Notes — InkEngine v3.3 ("Correct & Safe" — Long-Context Parity, Security Hardening & Async Race Fixes)

> **Suggested PR Title:** `v3.3: Fix SWA ring wrap, RAM pinning, silent global-KV page-table race, harden local HTTP API auth/timeouts/overflow/UTF-8, and add --verify-suite long-context parity gate`

This release implements **Milestone 1 ("Correct & Safe")** of the senior-agent architecture audit (`R0`, `R1`, `R2`, `R3`, `R5`, `R7`, `R9`), hardening correctness past position `1,024`, eliminating local HTTP security & stability hazards, and adding a permanent automated `--verify-suite` long-context regression gate — which immediately uncovered and fixed **3 deeper GPU/CPU concurrency & coherency bugs** (including a silent cross-layer Global-KV page-table race).

---

## Summary of Fixes & Improvements (v3.3)

### 1. Eliminated Batched SWA Ring Overwrite Past `pos = 1,024` (`R1 / SR3-01` — `src/paged_kv.h`, `src/paged_kv.cpp`, `src/cuda_kernels.cu`)
- **Root Cause**: Sliding-Window Attention (SWA) ring buffers previously allocated exactly `swa_window = 1,024` slots. During batched prefill (`B <= 128`) or speculative verification (`K <= 6`) past position `1,024`, writing the whole chunk before attention overwrote `B` oldest slots still inside query `0`'s `1,024`-token window.
- **Fix**:
  - Expanded `swa_ring_cap_ = swa_window_ + 128 = 1,152` slots (`+5.6 MiB` VRAM total at `Q4_0`: `45.0 → 50.6 MiB`; `160.0 → 180.0 MiB` at `F16`) and indexed SWA writes/reads modulo `swa_ring_cap_`, guaranteeing zero overlap for any batch up to `B = 128`.
  - Bumped the persistent NVMe KV checkpoint format `v2 → v3` (`hdr.version = 3u`) so legacy 1,024-slot checkpoints auto-invalidate and rebuild cleanly.

### 2. Fixed Cold Expert RAM Slot Eviction Before Read During Prefill (`R5 / SR3-04` — `src/expert_cache.h`, `src/expert_cache.cpp`, `src/model.cpp`)
- **Root Cause**: Under tight RAM budgets where not all cold experts fit in RAM simultaneously, loading a later micro-batch expert during `forward_batch` could evict an earlier expert's RAM slot before the GPU/CPU finished reading it.
- **Fix**: Added `pin_ram_expert()` / `unpin_ram_slots()` ref-counted pinning in `ExpertCache` and micro-batch pinning around batched MoE execution in `forward_batch()`.

### 3. Hardened Local HTTP API Security & OpenCode Auto-Auth (`R2 / SR1-09` — `src/server.h`, `src/server.cpp`, `src/vision.h`, `src/vision.cpp`, `src/main.cpp`, `inkcode.ps1`)
- **Bearer Token Auth by Default**: Generates a CSPRNG token (`--api-key <tok>` to set explicitly, `--no-auth` to opt out) and enforces `Authorization: Bearer <token>` across **all** endpoints — including `/health`, `/stats`, and `/shutdown`.
- **Blocked DNS Rebinding & Wildcard CORS**: Removed `Access-Control-Allow-Origin: *`, validated the HTTP `Host` header against local loopback allowlist (returning `403` on mismatched hosts), rejected cross-origin requests by default (`403`), and added opt-in `--cors-origin <origin>`.
- **Blocked Local / UNC Image Exfiltration Over HTTP**: `VisionEncoder::encode()` now defaults to `allow_local_files = false` over HTTP (blocking raw local file paths, `file:///`, and UNC `\\host\share` / device paths with HTTP `400`), while keeping CLI `--image` local files working (`allow_local_files = true`) with UNC/device paths always blocked everywhere.
- **Seamless `inkcode.ps1` Token Sync**: Automatically creates a per-user CSPRNG token at `%LOCALAPPDATA%\InkEngine\api_token` and syncs strictly `provider.inkengine.options.apiKey` in `%USERPROFILE%\.config\opencode\opencode.jsonc` without touching any other configured providers.

### 4. Context Overflow Protection & Accurate `finish_reason: "length"` (`R3 / SR1-02` — `src/model.h`, `src/model.cpp`, `src/server.cpp`)
- Prompts with `prompt_ids.size() >= max_ctx` now return HTTP `400` with standard OpenAI `error.code = "context_length_exceeded"` instead of overflowing KV structures.
- Clamps `max_tokens` to remaining context capacity and propagates `GenResult::hit_max_ctx` so OpenAI responses accurately emit `"finish_reason": "length"`.

### 5. Split Multi-Byte UTF-8 Buffering in SSE Streaming (`R7 / SR1-12` — `src/server.cpp`)
- Added `Utf8StreamBuffer` independently for both `reasoning_content` and `content` streams, holding incomplete trailing UTF-8 byte sequences across token boundaries until complete (and flushing safely at end-of-stream) so emojis and non-ASCII characters never emit broken `\uFFFD` chunks.

### 6. Socket Timeouts, Anti-Smuggling Parser & Concurrent Worker Threads (`R9 / SR1-10` — `src/server.cpp`)
- Added `SO_RCVTIMEO = 10s`, `SO_SNDTIMEO = 30s`, and a 10s HTTP header deadline so stalled/slowloris TCP connections can never hang the server.
- Replaced naive header parsing with a strict line-by-line parser that explicitly rejects conflicting `Transfer-Encoding` + `Content-Length` smuggling requests (`400`).
- Handled connections on bounded per-connection worker threads (up to 16 concurrent sockets), acquiring `gen_mu` **only** around vision + inference so `/health` and `/stats` respond in `<1 ms` even when a slow client stalls a socket.

### 7. Permanent Long-Context (`>1,024`) Parity Regression Gate (`R0 / SR1-01` — `src/model.cpp`, `src/main.cpp`)
- Added `--verify-suite` covering four automated correctness gates:
  - **Gate 1A**: SWA ring-wrap Query-0 parity (`pos=1024 B=7` and `pos=1100 B=128` crossing the `1,152` ring boundary vs. scalar `B=1`).
  - **Gate 1B**: Post-wrap multi-chunk prefill parity across `pos [1024..1151]` (`B=128`, `B=32`, `B=7` vs. `B=1`).
  - **Gate 2**: Speculative decoding ON vs. OFF past `pos=1024` (`2,017`-token prompt, 96 greedy tokens from identical KV).
  - **Gate 3**: In-place Turn-Rewind (`1088→1089`) and Disk v3 Checkpoint save/restore parity (bit-exact `0.000000`).

### 8. Three Bonus Concurrency & Coherency Bugs Caught & Fixed by R0!
Running `--verify-suite` immediately surfaced and allowed us to eliminate three subtle bugs that only triggered on fast VRAM-resident paths:
1. **Silent Async Global-KV Page-Table Race (`src/paged_kv.cpp`)**: All 8 Global Attention layers (`il = 5, 11, 17, 23, 29, 35, 41, 47`) previously shared a single `h_page_k_ptrs_` / `d_page_k_ptrs_` buffer. When core layers ran 100% in VRAM (zero-sync fast path), the CPU launched all 8 global layers asynchronously in microseconds and overwrote `h_page_*` while earlier layers' `cudaMemcpyAsync` were still reading it, causing global layers to attend to other layers' KV pages. **Fix**: Disjoint per-global-layer host+device page tables (`8 × max_pages_per_layer_`, zero extra GPU syncs). Result: `Q4_0` Gate 2 jumped from `2/18` failure to **`96/96` exact (`77/77 = 100%` accepted)** and Gate 3 jumped to **`0.000000` BIT-EXACT**.
2. **Per-Thread System Memory Fence in `k_moe_split_hits` (`src/cuda_kernels.cu`)**: Previously only `threadIdx.x == 0` issued `__threadfence_system()` after all 256 threads wrote zero-copy pinned `d_mapped_ffn_in`. **Fix**: Every thread now calls `__threadfence_system()` before `__syncthreads()`.
3. **Write-Combining Coherency & Unaligned CPU MoE Slices (`src/model.cpp`, `src/paged_kv.cpp`, `src/expert_cache.cpp`, `src/vision.cpp`, `src/cpu_kernels.cpp`)**: Replaced `cudaHostAllocWriteCombined` with coherent `cudaHostAllocDefault` on host staging buffers (including `h_moe_cpu_out_`, which 10 CPU threads read-modify-write), aligned `run_moe_experts` / `run_moe_experts_grouped` row slices to 64-byte (16-float) cache lines, switched 4-byte stack `d_tau_global_` copies to synchronous `cudaMemcpy`, and synced at end-of-token when `need_logits == false`.

---

## Files Changed (v3.3)

| File | Changes |
|---|---|
| `src/paged_kv.h`, `src/paged_kv.cpp` | `swa_ring_cap_ = 1152` (+5.6 MiB Q4_0), Disk checkpoint `v2 → v3`, disjoint per-global-layer `[8 × max_pages]` page tables, `cudaHostAllocDefault` |
| `src/expert_cache.h`, `src/expert_cache.cpp` | `pin_ram_expert()` / `unpin_ram_slots()` ref-counted RAM slot pinning, `set_swaps_frozen()` for deterministic parity gates, `cudaHostAllocDefault` |
| `src/cuda_kernels.cu` | Per-thread `__threadfence_system()` in `k_moe_split_hits`, `swa_ring_cap` modular indexing |
| `src/cpu_kernels.cpp` | 64-byte (16-float) cache-line aligned slices in `run_moe_experts` and `run_moe_experts_grouped` |
| `src/vision.h`, `src/vision.cpp` | `allow_local_files = false` default (HTTP safe), UNC/device path blocking everywhere, `cudaHostAllocDefault` |
| `src/model.h`, `src/model.cpp` | `run_parity_suite()` (`--verify-suite`), `GenResult::hit_max_ctx`, micro-batch RAM pinning in `forward_batch()`, `cudaHostAllocDefault`, safe `d_tau_global_` copy & `!need_logits` sync, v3.3 banner |
| `src/server.h`, `src/server.cpp` | CSPRNG Bearer auth on all routes, `Host` header check, no wildcard CORS + `--cors-origin`, block local `image_url`, HTTP 400 `context_length_exceeded` + `"length"` `finish_reason`, `Utf8StreamBuffer`, socket timeouts + strict anti-smuggling parser + 16-worker concurrency |
| `src/main.cpp` | Added `--verify-suite`, `--api-key`, `--no-auth`, `--cors-origin`; CLI `--image` enables `allow_local_files=true` |
| `inkcode.ps1` | Auto-creates `%LOCALAPPDATA%\InkEngine\api_token`, safely syncs only `provider.inkengine.options.apiKey` in `opencode.jsonc`, passes `--api-key` and Bearer headers on `/health`, `/stats`, `/shutdown` |
| `README.md`, `CHANGELOG.md`, `INKENGINE_OVERVIEW.txt` | Updated all documentation for v3.3 |

---

## Verification (v3.3)

- **Build**: Clean compile with `build.bat` (`sm_89`, CUDA 12.8 + MSVC x64 AVX2, zero warnings/errors).
- **Test A — Full `F16` Golden + R0 Parity Suite (`--ctx 4096 --kv-quant f16 --no-fp16-loop --verify --verify-suite`) — `exit 0` (ALL GATES PASSED)**:
  - Golden `--verify`: `29/29` tokenizer match, `"17 * 23 is 391."` (**100% EXACT MATCH**).
  - Gate 1A: `pos=1024 B=7[0] vs B=1`: `L0_SWA_err=0.000001 | top1=2623 MATCH | cos=0.989895`; `pos=1100 B=128[0]` crossing `1,152` wrap: `L0_SWA_err=0.015070` (**PASS**).
  - Gate 1B (`pos [1024..1151]`): `B=128` (`cos=0.998532, mean|d|=0.22962`), `B=32` (`cos=0.999458, mean|d|=0.12473`), `B=7` (`cos=0.999126, mean|d|=0.15095`) — all `top1=152914 MATCH` (**PASS**).
  - Gate 2 (`2,017`-tok prompt, 96 greedy tokens): `spec_off=96 | spec_on=96 (accepted=77/77 = 100%) | exact=YES` (**PASS**).
  - Gate 3: Turn-Rewind (`1088→1089`) **`max|dlogit|=0.000000` (BIT-EXACT)**; Disk v3 Checkpoint (`1088→1089`) **`max|dlogit|=0.000000` (BIT-EXACT)**.
- **Test B — Production `Q4_0` R0 Parity Suite (`--ctx 4096 --kv-quant q4 --verify-suite`) — `exit 0` (ALL GATES PASSED)**:
  - Gate 1A: `pos=1024 B=7[0]`: `L0_SWA_err=0.000001 | top1=2623 MATCH | cos=0.994599`; `pos=1100 B=128[0]`: `L0_SWA_err=0.014219` (**PASS**).
  - Gate 1B: `B=128` (`cos=0.998255, mean|d|=0.21061`), `B=32` (`cos=0.993900, mean|d|=0.39767`), `B=7` (`cos=0.994518, mean|d|=0.39723`) — all `top1=152914 MATCH` (**PASS**).
  - Gate 2: `spec_off=96 | spec_on=96 (accepted=77/77 = 100%) | exact=YES` (**PASS**).
  - Gate 3: Turn-Rewind **`max|dlogit|=0.000000` (BIT-EXACT)**; Disk v3 Checkpoint **`max|dlogit|=0.000000` (BIT-EXACT)**.
- **Test C — Tight-RAM R5 Pinning Stress (`--ctx 4096 --ram-mb 96 --no-preload-ram --kv-quant f16 --no-fp16-loop --verify`) — `exit 0`**:
  - Only `29` RAM slots (`93.4 MiB`), `"17 * 23 is 391."` (**100% EXACT MATCH**).
- **Test D — Live HTTP Security, Timeout, Overflow, UTF-8 & Vision Suite (15/15 PASS)**:
  - Missing/wrong Bearer token & unauthenticated `/shutdown` → `401`; authenticated `/health` → `200`, no wildcard CORS.
  - Cross-origin `OPTIONS` & DNS-rebinding `Host: evil.attacker.com:8799` → `403`; conflicting `Transfer-Encoding` + `Content-Length` → `400`.
  - Local path, `file:///`, and UNC `\\127.0.0.1\c$\test.png` HTTP `image_url` → all blocked (`400`).
  - `/health` during a stalled partial-header TCP connection → **`200` in `0.6 ms`**.
  - Prompt `>= max_ctx` → `400` with `error.code = "context_length_exceeded"`.
  - Multi-byte emoji SSE stream → `invalid=0`, 12 SSE chunks, exact `'🚀 🔥 🎉 🌊 🧠'`.
  - Authenticated `data:image/png;base64` vision → `200`, accurate `'INKENGINE 3.2'`; authenticated `/shutdown` → `200`, clean exit.
- **Test E — `inkcode.ps1` End-to-End Flow (`start` → token & `opencode.jsonc` check → `status` → `stop`)**:
  - Auto-created `%LOCALAPPDATA%\InkEngine\api_token` (`ink-...`), synced `provider.inkengine.options.apiKey` in `opencode.jsonc` while leaving all other providers (`cerebras`, `modal`, `tokenrouter`) completely untouched.
  - `inkcode status` reported `Vision=ON (0 MiB idle)`, `SWA 50.6 MiB` (`KV=723.2 MiB`, `Experts=4,838.2 MiB`), and `inkcode stop` cleanly freed all VRAM and RAM.

---

# Changelog / PR Notes — InkEngine v3.2 (Native Multimodal Vision — Zero Permanent VRAM/RAM)

> **Suggested PR Title:** `v3.2: Add standalone GPU-streamed inkling2v multimodal vision encoder (0 MiB permanent VRAM/RAM), OpenAI image_url API, and OpenCode image support`

This release adds **100% standalone native multimodal vision** (`mmproj-F16.gguf`, `projector_type = inkling2v`) to InkEngine — enabling Inkling-2-Mini to directly inspect screenshots, UI renders, Three.js canvases, diagrams, and photos inside OpenCode and the OpenAI HTTP API with **zero permanent VRAM increase**, **zero permanent RAM increase**, and **0% slowdown on text-only workloads**.

---

## Architectural Decision: On-Demand GPU Tensor-Core Streaming vs. Permanent CPU RAM

We evaluated two architectures for the 610 MB (`318M`-parameter) `mmproj-F16.gguf` vision encoder on an 8 GB VRAM / 16 GB RAM system:

| Approach | Permanent VRAM | Permanent RAM | HD Image Encode Time | Impact on Text Decode |
|---|---:|---:|---:|---|
| Pin in VRAM | +610 MiB *(loses ~185 hot MoE expert slots!)* | 0 MiB | ~0.2s | **Slower text decode** (fewer VRAM experts) |
| Pin in CPU RAM (AVX2) | 0 MiB | +610 MiB *(evicts ~185 RAM experts to SSD!)* | ~5–15s | **Triggers SSD paging on 16 GB RAM** |
| **On-Demand GPU Tensor-Core Streaming (Chosen)** | **0 MiB** | **0 MiB** *(<100 KB GGUF header)* | **~3.0s** *(SATA stream)* / **<0.5s** *(NVMe / OS cache)* / **0.0 ms** *(LRU hit)* | **0.0% impact — all 1,494 VRAM + RAM expert slots untouched!** |

By streaming `mmproj-F16.gguf` weights layer-by-layer in **8–9 MiB slices** through the engine's **already-allocated 14 MiB PCIe prefill staging ring** (`d_prefill_stage_[0]` + `h_prefill_stage_[0]`), reusing the idle `Q/K/V` activation slab for single-pass FFN, and freeing all transient activation VRAM (`cudaFree`) immediately after each image, vision adds **literally zero steady-state memory footprint**.

---

## Summary of New Features (v3.2)

### 1. Six Custom Multimodal Vision CUDA Kernels (`src/cuda_kernels.cuh`, `src/cuda_kernels.cu`)
- `ink_cuda_vision_layernorm`: Exact mean + variance LayerNorm (`eps = 1e-5`) across warp-convergent 256-thread blocks.
- `ink_cuda_vision_gemm`: WMMA FP16→FP32 Tensor Core GEMM supporting `F16`, `BF16`, and `F32` weights + optional bias + custom output stride (`ld_out`), reusing existing `g_batch_f16_buf` (`0` new persistent VRAM).
- `ink_cuda_vision_rope_2d`: Exact 2D Vision RoPE (`pos_w` on dims `[0..32)`, `pos_h` on dims `[32..64)`, adjacent pairs `mode=0`, `theta=10000.0`, CLS at `(0,0)`, patches 1-indexed `(x+1, y+1)`).
- `ink_cuda_vision_flash_attn`: Tiled bidirectional FlashAttention (`d_head=64`, O(1) VRAM).
- `ink_cuda_vision_gelu_erf` & `ink_cuda_vision_layerscale_add`: Exact `GELU_ERF` activation and residual LayerScale fusion.

### 2. Standalone Zero-Dependency `VisionEncoder` (`src/vision.h`, `src/vision.cpp`)
- **Pure Windows WIC Decoder** (`windowscodecs.lib` + `ole32.lib`): Decodes PNG, JPEG, BMP, GIF, and WebP directly from memory (`data:image/...;base64,...`, `file:///`, bare base64, or local file paths) into 24-bit RGB without `stb_image` or external libraries.
- **Exact `inkling2v` Pipeline**: Dynamic VRAM-safe pixel budgeting (caps 8 GB GPUs at `640` tokens = crisp `1320×720` / `1280×720` HD for fast single-pass FFN & LLM prefill), align-40 bottom-right `-1.0f` padding, `(v - mean*255)/(std*255)` normalization, `20×20` patch conv2d GEMM, bilinear resized `32×32` learned `position_embd`, `v.pre_ln`, 23 ViT layers streamed in 3 small slices/layer, exact 2×2 spatial merge `[r0c0, r0c1, r1c0, r1c1]`, and streamed `mm.1` (`4096→4096`) + `GELU_ERF` + `mm.2` (`4096→2304`) projector.
- **On-Demand Content-Hash LRU Cache**: Caches up to 16 encoded images (capped at `64 MiB` total RAM, `0 MiB` until images are actually sent) keyed by 64-bit FNV-1a content hash so multi-turn follow-ups skip re-encoding completely (`0.0 ms`).

### 3. Engine Multimodal Injection & Automatic KV Prefix Caching (`src/gguf.*`, `src/tokenizer.*`, `src/model.*`, `src/main.cpp`)
- Auto-detects `mmproj-F16.gguf` / `BF16` / `F32` beside the model GGUF (overridable via `--mmproj <path>` or disabled via `--no-vision`; standalone CLI testing via `--image <path>`).
- Implements Inkling-2's official multimodal chat template (`<|content_image|><|unused_200054|><|end_message|>` for both `user` and `tool` messages) and **bypasses `tok_norm_`** for raw vision embeddings in `forward_batch()` (matching official `inkling2.cpp` spec).
- Expands image placeholders into deterministic **negative synthetic token IDs** derived from each image's 64-bit hash — allowing both the incremental KV prefix cache and `TurnRewindSnapshot` to automatically cache and restore prior images in **`~10 ms`** across multi-turn conversations!

### 4. OpenAI Vision API & OpenCode Integration (`src/server.cpp`, `inkcode.ps1`, `opencode.jsonc`, `build.bat`)
- Parses OpenAI `image_url` (`{"type":"image_url","image_url":{"url":"..."}}`), `input_image`, and Anthropic-style base64 `image` content blocks in `user` and `tool` messages.
- Reports `"vision": true|false` in `GET /health` and `Vision=ON (0 MiB idle)` in `inkcode status`; adds `--mmproj` / `--no-vision` passthrough to `inkcode.ps1`.
- Enabled `"modalities": { "input": ["text", "image"], "output": ["text"] }` in OpenCode's `opencode.jsonc` provider config so OpenCode can attach screenshots directly to `Inkling-2-Mini`.

---

## Files Changed (v3.2)

| File | Changes |
|---|---|
| `src/vision.h`, `src/vision.cpp` | *(New)* Standalone WIC decoder, GPU-streamed `inkling2v` ViT + projector, FFN `d_work` buffer reuse, content-hash LRU cache |
| `src/cuda_kernels.cuh`, `src/cuda_kernels.cu` | Added 6 vision CUDA kernels (LayerNorm, WMMA FP16→FP32 GEMM, 2D RoPE, tiled FlashAttention `d=64`, GELU-ERF, LayerScale) |
| `src/gguf.h`, `src/gguf.cpp` | Added `get_f32_arr()` metadata helper for `clip.vision.image_mean` / `image_std` |
| `src/tokenizer.h`, `src/tokenizer.cpp` | Added `ChatContentPart`, `ChatMessage::parts`, `image_placeholder_id() == 200054`, official `<|content_image|>` chat/tool formatting |
| `src/model.h`, `src/model.cpp` | Auto-load `mmproj*.gguf`, `MultimodalEmbeddings`, `expand_multimodal_tokens()`, `tok_norm_`-bypassing `raw_embd` in `forward_batch()`, homogeneous image/text prefill chunking, v3.2 banner |
| `src/server.cpp` | Multimodal `image_url` / `input_image` / `image` parsing, `/health` `vision` field, HTTP 400 validation, `imgs=%zu` logging |
| `src/main.cpp` | Added `--mmproj`, `--no-vision`, and repeatable `--image` CLI flags |
| `inkcode.ps1` | Added `--mmproj` / `--no-vision` / `INKENGINE_MMPROJ` support and `Vision=ON (0 MiB idle)` telemetry |
| `build.bat` | Added `src/vision.cpp` and Windows system libs `windowscodecs.lib ole32.lib` |
| `README.md`, `CHANGELOG.md`, `INKENGINE_OVERVIEW.txt` | Updated all documentation for v3.2 multimodal vision |

---

## Verification (v3.2)

- **Build**: Clean compile with `build.bat` (`sm_89`, CUDA 12.8 + MSVC x64 AVX2, zero warnings/errors).
- **Zero Permanent VRAM / RAM Overhead (`inkcode start --ctx 131072`)**:
  - Expert VRAM pool remained **identically `1,494` dynamic slots (`4,841.6 MiB`)** as v3.1 (`Vision=ON (0 MiB idle)`).
  - GPU VRAM measured via `nvidia-smi`: `7,669 MiB` idle → `7,730 MiB` after 640×480 → `7,714 MiB` after Full HD 1920×1080 (**0 MiB VRAM leak**, all transient activation buffers freed immediately).
- **Real Image Recognition (CLI `--image` + Live OpenAI Base64 Streaming)**:
  - **640×480 synthetic test PNG**: Inkling-2-Mini accurately read `"INKENGINE 3.2"` (black pixelated text at top) and identified the red circle (upper-left), blue square (upper-right), and green triangle (bottom-center, pointing up).
  - **Multi-turn follow-up (same base64 image in history)**: **0.0 ms vision LRU cache hit** + **`192` cached KV tokens restored in `10.0 ms`** via turn-rewind (`TTFT 19.9s → 2.0s`), answered `"Blue square"` accurately.
  - **Full HD `1920×1080` PNG (`→ 1320×720`, `2,376` patches → `594` LLM tokens)**: Encoded in **`2.99s`** streamed straight from SATA `S:\`, prefilled at `74.1 tok/s`, accurately read `"INKENGINE 3.2"` and all three shapes.
- **Text-Only Exact Parity (`--ctx 4096 --kv-quant f16 --no-fp16-loop --verify`)**:
  - `29/29` tokenizer match, `"17 * 23 is 391."` — **100% EXACT MATCH** vs. `llama.cpp`.

---

# Changelog / PR Notes — InkEngine v3.1 (Post-v3.0 Stability, Quality & Streaming Fixes)

> **Suggested PR Title:** `v3.1: Fix Q4_0 KV bias & speculative greedy-loop bugs, official sampling defaults, SSE heartbeats, <1s disconnect detection, and ~4ms turn-rewind KV cache`

This release fixes **6 high-impact quality, streaming, and agentic-reliability issues** discovered during long multi-turn OpenCode sessions after the initial `v3.0` release (`bc12abc`).

---

## Summary of Fixes & Improvements

### 1. Eliminated Systematic `+0.5d` Bias in `Q4_0` KV Cache Quantization (`src/cuda_kernels.cu`, `src/paged_kv.cpp`)
- **Root Cause**: `quant_block32_q4_0` used `__float2int_rn(v * id + 8.5f)`. Adding `+ 8.5f` is only valid for truncating `(int)` casts — combining `+ 8.5f` with round-to-nearest (`__float2int_rn`) double-shifted every quantized value by `+0.5 * d`, injecting a systematic DC bias into every K and V vector across all 48 logical layers and doubling worst-case `Q4_0` KV error. On extended reasoning chains (`effort=max`), this caused corrupted code snippets and repetitive loops.
- **Fix**:
  - Changed rounding to exact unbiased `__float2int_rn(v * id + 8.0f)` (halving max `Q4_0` KV error and preserving exact zero).
  - Bumped the persistent NVMe KV checkpoint format from `v1` → `v2` (`hdr.version = 2u`) so stale pre-fix `Q4_0` `.ckpt` files on disk are automatically invalidated and rebuilt cleanly.

### 2. Fixed Greedy-Argmax Trap on Speculative Rejection (`src/model.cpp`)
- **Root Cause**: In `sample_from_logits`, when the N-gram speculative drafter proposed a token that was rejected and did not appear in the filtered candidate set (`d_idx < 0`), execution returned `cand[0].second` (pure greedy `temp=0` argmax) instead of sampling from the residual distribution `p'(x) = p(x)`. Once a 3-gram phrase began repeating, the drafter fired on every step and rejections repeatedly collapsed sampling to greedy argmax, locking repetitions into a loop.
- **Fix**:
  - Fixed rejection when `d_idx < 0` to fall through to exact unbiased random sampling from `cand` (restoring mathematically exact Leviathan et al. lossless speculative sampling).
  - Added fast-path unsorted sampling when `top_k == 0 && top_p >= 1.0f` and switched softmax/CDF normalization to `double` precision for numerical stability.

### 3. Aligned Default Sampling Parameters with Official Inkling-2-Mini Spec (`src/server.cpp`, `src/model.h`)
- **Root Cause**: HTTP server defaults were previously `temperature = 0.6`, `top_p = 0.95`, `top_k = 40`, `min_p = 0.05` (with a fixed deterministic seed sequence starting at `42`). Over-sharpening logits at `T=0.6` with `min_p=0.05` exacerbated reasoning loops under high/max effort.
- **Fix**:
  - Updated HTTP server defaults to Thinking Machines / Unsloth's official Inkling-2-Mini parameters: **`temperature = 1.0`**, **`top_p = 1.0`**, **`top_k = 0`**, **`min_p = 0.0`**, with a time-mixed default RNG seed (while still honoring explicit client `"seed"` values).
  - Updated `SamplingParams` struct defaults in `src/model.h` (`top_k = 0`, `top_p = 1.0f`, `min_p = 0.0f`) so CLI `--temp` runs match.
  - Added live sampling telemetry (`T=1.00 p=1.00 k=0 mp=0.00`) to `[HTTP] Incoming` server logs.

### 4. Real OpenAI SSE `data:` Heartbeats During Large Tool Calls & Prefills (`src/server.cpp`)
- **Root Cause**: While Inkling-2 generated large file-write tool calls (`ST_TOOL_JSON` / `ST_TOOL_TEXT`), InkEngine buffered the full JSON silently without emitting any bytes over TCP for several minutes — causing OpenCode / Vercel AI SDK clients to fire `SSE read timed out`. Similarly, `: keepalive\n\n` SSE comments during long prefills did not reset chunk timers that expect `data:` events.
- **Fix**:
  - Added `emit_sse_heartbeat()` emitting valid OpenAI empty-delta SSE events (`data: {...,"delta":{}, "finish_reason":null}\n\n`) on every prefill batch chunk (~1.5s) and every 16 tokens (~0.7s) while buffering `ST_TOOL_JSON`, `ST_TOOL_TEXT`, or `ST_HEADER`.
  - Fixed `finish_reason` to accurately report `"length"` when `gen_tokens >= max_tokens` instead of always `"stop"`.

### 5. Instant `<1s` TCP Disconnect Detection — Eliminates Zombie Blocking & `ProviderHeaderTimeoutError` (`src/server.cpp`)
- **Root Cause**: If a client aborted, timed out, or pressed `Ctrl+C` while InkEngine was silently buffering a tool call, InkEngine never noticed the closed socket and kept burning GPU for 5–12+ minutes on a dead connection — blocking the single-threaded HTTP server loop so queued retries hit `ProviderHeaderTimeoutError: 300000ms`.
- **Fix**:
  - Added non-blocking `<1 µs` Winsock2 `is_client_alive()` (`select` + `recv(..., MSG_PEEK)` detecting TCP `FIN`/`RST`), polled on every prefill batch and every 8 decoded tokens.
  - Immediately aborts dead requests, logs `ABORTED(client_closed)`, skips writing trailing SSE chunks to closed sockets, and frees the server for incoming requests in `<1s`.

### 6. Lightweight `~49 MiB` In-Place Turn-Level KV Rewind Cache (`src/paged_kv.h`, `src/paged_kv.cpp`, `src/model.h`, `src/model.cpp`)
- **Root Cause**: When a client retried or canceled a large turn (e.g., 30K–36K tokens) after partial generation (`kv_tokens_.size() >= prompt_tokens.size()`), the incremental `P > L` check missed, forcing a fallback to the 11K system checkpoint and a 5+ minute re-prefill of 25K tokens.
- **Fix**:
  - Added `TurnRewindSnapshot` (`save_turn_snapshot` / `rewind_to_turn_snapshot`) tracked by a `reset_epoch_` counter in `HybridPagedKV`.
  - Snapshots **only ShortConv + SWA ring state (`~49 MiB`, zero Global KV page copying)** at the prompt's 64-token page boundary (and every 4,096 tokens during long prefills).
  - On retries/cancels, rewinds in-place in **`~4 ms`** while keeping all Global KV pages `[0..snap_pages)` intact across VRAM, RAM, and NVMe SSD (scales cleanly to 128K–512K+ context!).

### 7. Updated OpenCode Configuration Example (`README.md`)
- Added `"timeout": false`, `"headerTimeout": false`, and `"chunkTimeout": false` to the `opencode.jsonc` provider options example so OpenCode never aborts long local prefills or large multi-thousand-token generations at its default 5-minute (`300000 ms`) timeout.

---

## Files Changed

| File | Changes |
|---|---|
| `src/cuda_kernels.cu` | Fixed `quant_block32_q4_0` rounding bias (`+ 8.5f` → `+ 8.0f`) |
| `src/paged_kv.h` | Added `TurnRewindSnapshot`, `reset_epoch_`, `save_turn_snapshot`, `rewind_to_turn_snapshot` |
| `src/paged_kv.cpp` | Bumped disk KV checkpoint version `v1 → v2`; implemented ~4 ms zero-Global-copy in-place turn rewind |
| `src/model.h` | Aligned `SamplingParams` defaults (`top_k=0, top_p=1.0, min_p=0.0`); added `turn_snap_` member |
| `src/model.cpp` | Fixed speculative rejection greedy-argmax bug; double-precision unsorted sampling fast path; integrated turn-rewind save/restore |
| `src/server.cpp` | Official Inkling-2 sampling defaults (`T=1.0, p=1.0, k=0, mp=0.0`); time-mixed seed; `is_client_alive()` `<1s` disconnect detection; SSE `data:` heartbeats during prefill & `ST_TOOL_JSON`; accurate `"length"` `finish_reason` |
| `README.md` | Added OpenCode local timeout disable options (`timeout`/`headerTimeout`/`chunkTimeout: false`) |

---

## Verification

- **Build**: Clean compile with `build.bat` (`sm_89`, CUDA 12.8 + MSVC x64 AVX2, zero errors).
- **Quality / Loop Regression (`Q4_0` KV + Speculative ON + `effort=max` + `T=1.0`)**:
  - Same Python `LRUCache` prompt that previously looped and emitted duplicated pointer lines inside reasoning at `2,500` tokens now completes coherent reasoning and emits clean, bug-free Python code at **`25.13 tok/s`** (`spec=989/2879`).
- **SSE Heartbeats + `<1s` Disconnect + Retry KV Reuse + File-Write Tool Call**:
  - Abruptly closing a client socket mid-prefill (`2,112 / 2,127` tokens) after `16` SSE `data:` heartbeats triggered immediate abort (`ABORTED(client_closed)`) and `/health` responded in **`842.1 ms`**.
  - Immediately retrying the identical `2,127`-token prompt reused **`2,112 / 2,127` cached tokens (`99.3%`)**, emitted **`18` SSE `data:` heartbeats** during `write_file` JSON buffering, and streamed the complete `tool_calls` + `[DONE]` response at **`24.39 tok/s`**.
