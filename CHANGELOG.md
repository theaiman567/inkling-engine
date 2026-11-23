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
