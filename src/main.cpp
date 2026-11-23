#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include "model.h"
#include "server.h"

namespace {

void print_usage() {
    std::printf(
        "InkEngine - Custom Looped-MoE 3-Tier (VRAM/RAM/SSD) Serving Engine for Inkling-2-Mini\n\n"
        "Usage: inkengine.exe [options]\n"
        "  --model <path>         Path to Inkling-2-Mini GGUF file\n"
        "  --mmproj <path>        Path to mmproj-F16.gguf vision projector (auto-detected beside model GGUF)\n"
        "  --no-vision            Disable multimodal vision encoder\n"
        "  --image <path>         Attach an image file (PNG/JPG/BMP/WebP) to --prompt (repeatable)\n"
        "  --ctx <N>              Max context length (default: 65536; supports up to 1048576 via SSD!)\n"
        "  --kv-quant <q4|q8|f16> KV cache quantization (default: q4)\n"
        "  --kv-mode <auto|quest|exact> Global KV mode past VRAM (default: auto)\n"
        "  --quest-pages <N>      Top historical pages for Quest attention (default: 128 = 8192 tok)\n"
        "  --kv-swap-dir <dir>    Directory for SSD KV swap file (default: %%TEMP%% on NVMe C:)\n"
        "  --vram-kv-mb <N>       VRAM budget for global KV cache in MiB (0 = auto)\n"
        "  --ram-kv-mb <N>        RAM budget for Tier-1 KV cache in MiB (default: 1024)\n"
        "  --ram-mb <N>           RAM budget for Tier-1 experts in MiB (0 = auto; low value tests SSD paging)\n"
        "  --policy <loop-lfu|core-pinned> Expert VRAM cache policy (default: loop-lfu)\n"
        "  --no-preload-ram       Demand-page non-VRAM experts directly from SSD on first touch\n"
        "  --no-fp16-loop         Keep loop controller weights in F32 instead of FP16\n"
        "  --predictor <path>     Path to main .moepred expert predictor (auto-detected by default)\n"
        "  --predictor-start <p>  Path to start .moepred expert predictor (auto-detected by default)\n"
        "  --no-predictor         Disable learned MoE expert predictor (fall back to loop-lfu)\n"
        "  --no-graphs            Disable CUDA Graphs for layer preludes/postludes\n"
        "  --no-batch-prefill     Disable Batched GEMM + Grouped MoE prefill (use 1-tok prefill)\n"
        "  --no-speculative       Disable zero-VRAM N-gram / prompt-lookup speculative decoding\n"
        "  --no-self-draft        Disable 1-Loop (24L) GPU-Only lossless self-speculative drafting\n"
        "  --self-draft-k <N>     Opt-in 1-Loop (24L) GPU-Only self-draft tokens (0 = off [default], 2-3 = code turbo)\n"
        "  --no-disk-kv-cache     Disable persistent NVMe OpenCode KV prefix checkpoint cache\n"
        "  --probe-early-exit     Empirically probe 1-Loop (24L) vs Full 3-Loop (48L) agreement\n"
        "  --verify-predictor     Verify native MoE predictor against PyTorch golden .vec test vectors\n"
        "  --bench-prefill <N>    Benchmark batched prompt prefill speed on N tokens\n"
        "  --threads <N>          CPU AVX2 worker threads for missed experts (default: 10)\n"
        "  --verify               Run golden correctness check vs llama.cpp reference tokens\n"
        "  --verify-suite         Run R0 long-context (>1024 SWA), speculative & checkpoint parity gate suite\n"
        "  --bench <N>            Benchmark decode throughput for N tokens (with optional --bench-pos P)\n"
        "  --bench-pos <P>        Simulate long context at position P (e.g. 32768, 65536, 131072, 262144)\n"
        "  --profile              Print per-stage millisecond breakdown during --bench / --prompt\n"
        "  --prompt <text>        Run single prompt completion\n"
        "  --effort <level>       Thinking effort: none|minimal|low|medium|high|xhigh|max (default: max for --serve, none for --prompt)\n"
        "  --max-tokens <N>       Max output tokens (default: 256)\n"
        "  --temp <F>             Sampling temperature (default: 0.0 = greedy)\n"
        "  --serve                Start OpenAI-compatible HTTP server\n"
        "  --host <ip>            Server bind host (default: 127.0.0.1)\n"
        "  --port <P>             Server port (default: 8080)\n"
        "  --api-key <token>      Bearer auth token for HTTP server (default: auto-generated CSPRNG token)\n"
        "  --no-auth              Disable HTTP Bearer authentication (NOT recommended on shared machines)\n"
        "  --cors-origin <orig>   Allow CORS for one explicit origin (default: CORS disabled)\n"
    );
}

void print_profile(const InklingEngine& eng) {
    auto p = eng.profile();
    auto es = eng.expert_stats();
    auto ks = eng.kv_stats();
    if (p.tokens > 0) {
        double n = (double)p.tokens;
        std::printf("\n--- Per-Token Profile Breakdown (avg over %d tokens) ---\n", p.tokens);
        std::printf("  Embed + TokNorm : %6.2f ms\n", p.embed_ms / n);
        std::printf("  AttnRes + Loop  : %6.2f ms\n", p.mix_loop_ms / n);
        std::printf("  Attention + Conv: %6.2f ms\n", p.attn_ms / n);
        std::printf("  MoE Router      : %6.2f ms\n", p.router_ms / n);
        std::printf("  MoE Core (36L)  : %6.2f ms\n", p.moe_core_ms / n);
        std::printf("  MoE Outer (12L) : %6.2f ms\n", p.moe_outer_ms / n);
        std::printf("  LM Head + Argmax: %6.2f ms\n", p.lm_head_ms / n);
        std::printf("  TOTAL           : %6.2f ms  (%.2f tok/s)\n", p.total_ms / n, 1000.0 / (p.total_ms / n));
    }
    double hr  = es.total_lookups ? (100.0 * es.vram_hits / es.total_lookups) : 0.0;
    double chr = es.core_lookups  ? (100.0 * es.core_vram_hits / es.core_lookups) : 0.0;
    double ohr = es.outer_lookups ? (100.0 * es.outer_vram_hits / es.outer_lookups) : 0.0;
    std::printf("--- 3-Tier Memory Telemetry ---\n");
    std::printf("  Experts : VRAM hit=%.1f%% (Core=%.1f%%, Outer=%.1f%%) | RAM hits=%llu | SSD reads=%llu | DMA promos=%llu\n",
                hr, chr, ohr, (unsigned long long)es.ram_hits, (unsigned long long)es.ssd_reads, (unsigned long long)es.dma_promotions);
    std::printf("  PagedKV : pos=%d/%d | VRAM=%.1f MiB | RAM=%.1f MiB | SSD=%.1f MiB (reads=%llu, writes=%llu)\n",
                ks.cur_pos, ks.max_ctx, ks.swa_vram_mb + ks.global_vram_mb, ks.ram_kv_mb, ks.ssd_kv_mb,
                (unsigned long long)ks.ssd_page_reads, (unsigned long long)ks.ssd_page_writes);
}

} // namespace

int main(int argc, char** argv) {
    EngineConfig cfg;
    cfg.model_path = "S:\\Unsloth\\huggingface\\hub\\models--unsloth--Inkling-2-Mini-GGUF\\snapshots\\ae9327c5fbc7333116ba32eee78bb5d2bbefae4f\\Inkling-2-Mini-UD-IQ3_XXS.gguf";

    SamplingParams sp;
    bool do_verify = false;
    bool do_verify_suite = false;
    bool do_verify_pred = false;
    bool do_probe_ee = false;
    bool do_serve = false;
    bool do_profile = false;
    int  bench_n = 0;
    int  bench_pos = 0;
    int  bench_prefill_n = 0;
    std::string prompt_str;
    std::vector<std::string> image_paths;
    std::string effort = "";
    ServerConfig srv_cfg;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--help" || a == "-h") { print_usage(); return 0; }
        else if (a == "--model") cfg.model_path = next();
        else if (a == "--mmproj") cfg.mmproj_path = next();
        else if (a == "--no-vision") cfg.use_vision = false;
        else if (a == "--image") image_paths.push_back(next());
        else if (a == "--ctx") cfg.max_ctx = std::stoi(next());
        else if (a == "--kv-quant") {
            std::string q = next();
            if (q == "q4") cfg.kv_quant = KV_QUANT_Q4_0;
            else if (q == "q8") cfg.kv_quant = KV_QUANT_Q8_0;
            else if (q == "f16") cfg.kv_quant = KV_QUANT_F16;
        }
        else if (a == "--kv-mode") {
            std::string m = next();
            if (m == "auto") cfg.kv_mode = KV_ATTN_AUTO;
            else if (m == "quest") cfg.kv_mode = KV_ATTN_QUEST;
            else if (m == "exact") cfg.kv_mode = KV_ATTN_EXACT_STREAM;
        }
        else if (a == "--quest-pages") cfg.quest_top_pages = std::stoi(next());
        else if (a == "--kv-swap-dir") cfg.kv_swap_dir = next();
        else if (a == "--vram-kv-mb") cfg.vram_kv_mb = std::stoull(next());
        else if (a == "--vram-exp-mb") cfg.vram_exp_mb = std::stoull(next());
        else if (a == "--ram-kv-mb") cfg.ram_kv_mb = std::stoull(next());
        else if (a == "--ram-mb") cfg.ram_exp_mb = std::stoull(next());
        else if (a == "--policy") {
            std::string p = next();
            cfg.cache_policy = (p == "loop-lfu") ? CACHE_LOOP_LFU : CACHE_CORE_PINNED;
        }
        else if (a == "--no-preload-ram") cfg.preload_ram = false;
        else if (a == "--async-preload") cfg.async_ram_preload = true;
        else if (a == "--sync-preload") cfg.async_ram_preload = false;
        else if (a == "--no-fp16-loop") cfg.fp16_loop = false;
        else if (a == "--predictor") cfg.predictor_path = next();
        else if (a == "--predictor-start") cfg.predictor_start_path = next();
        else if (a == "--no-predictor") cfg.use_predictor = false;
        else if (a == "--no-graphs") cfg.use_cuda_graphs = false;
        else if (a == "--no-batch-prefill") cfg.use_batched_prefill = false;
        else if (a == "--no-speculative") cfg.use_speculative = false;
        else if (a == "--no-self-draft") cfg.self_draft_k = 0;
        else if (a == "--self-draft-k") cfg.self_draft_k = std::stoi(next());
        else if (a == "--no-disk-kv-cache") cfg.use_disk_kv_cache = false;
        else if (a == "--probe-early-exit") do_probe_ee = true;
        else if (a == "--verify-predictor") do_verify_pred = true;
        else if (a == "--bench-prefill") bench_prefill_n = std::stoi(next());
        else if (a == "--threads") cfg.n_threads = std::stoi(next());
        else if (a == "--verify") do_verify = true;
        else if (a == "--verify-suite") do_verify_suite = true;
        else if (a == "--bench") bench_n = std::stoi(next());
        else if (a == "--bench-pos") bench_pos = std::stoi(next());
        else if (a == "--profile") do_profile = true;
        else if (a == "--prompt") prompt_str = next();
        else if (a == "--effort") effort = next();
        else if (a == "--max-tokens") sp.max_tokens = std::stoi(next());
        else if (a == "--temp") sp.temperature = std::stof(next());
        else if (a == "--serve") do_serve = true;
        else if (a == "--host") srv_cfg.host = next();
        else if (a == "--port") srv_cfg.port = std::stoi(next());
        else if (a == "--api-key") srv_cfg.api_key = next();
        else if (a == "--no-auth") srv_cfg.require_auth = false;
        else if (a == "--cors-origin") srv_cfg.cors_origin = next();
    }

    if (do_verify_pred) {
        std::string pm = cfg.predictor_path.empty() ? "models\\inkling2-pred-v3.moepred" : cfg.predictor_path;
        std::string ps = cfg.predictor_start_path.empty() ? "models\\inkling2-start-v3.moepred" : cfg.predictor_start_path;
        MoePredictorModel m1, m2;
        bool ok1 = m1.load(pm, 24, 128, 2304) && m1.verify_vectors("models\\checks\\pred-v3.vec");
        bool ok2 = m2.load(ps, 24, 128, 2304) && m2.verify_vectors("models\\checks\\start-v3.vec");
        if (!do_verify && !do_serve && bench_n == 0 && prompt_str.empty()) {
            return (ok1 && ok2) ? 0 : 1;
        }
    }

    // Default to async background RAM preload in server mode (~14s boot time instead of ~34s!)
    if (do_serve && cfg.preload_ram) {
        bool explicit_sync = false;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--sync-preload") == 0) explicit_sync = true;
        }
        if (!explicit_sync) cfg.async_ram_preload = true;
    }

    InklingEngine engine;
    if (!engine.load(cfg)) {
        std::fprintf(stderr, "Failed to initialize InkEngine!\n");
        return 1;
    }

    if (do_verify) {
        std::printf("\n[VERIFY] Running exact token-parity test against llama.cpp reference...\n");
        // Exact golden prompt tokens captured from llama.cpp on this machine:
        std::vector<int32_t> golden_prompt = {
            200002,200004,34182,279,154554,150628,25,220,15,200010,200000,200004,
            1775,149884,220,68107,196987,220,68169,30,154710,149881,150011,150921,
            156021,13,200010,200001,200004
        };
        std::vector<int32_t> expected_out = { 68107, 196987, 220, 68169, 149884, 220, 89431, 13, 200010 };

        // Also test our custom tokenizer on the exact same chat prompt!
        std::string fmt = engine.tokenizer().format_chat({{"user", "What is 17 * 23? Answer in one short sentence."}}, "none", true);
        auto my_ids = engine.tokenizer().encode(fmt, true);
        std::printf("[VERIFY] Tokenizer check: %zu tokens (ref %zu) | exact_match=%s\n",
                    my_ids.size(), golden_prompt.size(), (my_ids == golden_prompt) ? "YES" : "NO");
        if (my_ids != golden_prompt) {
            std::printf("  my_ids : "); for (auto x : my_ids) std::printf("%d,", x); std::printf("\n");
            std::printf("  ref_ids: "); for (auto x : golden_prompt) std::printf("%d,", x); std::printf("\n");
        }

        SamplingParams gsp;
        gsp.temperature = 0.0f;
        gsp.max_tokens = 24;
        engine.reset_profile();
        auto res = engine.generate(golden_prompt, gsp, [&](int32_t id, const std::string& s) {
            std::printf("  gen tok=%d ('%s')\n", id, s.c_str());
            return true;
        }, do_profile);

        bool exact = (res.tokens == expected_out);
        std::printf("[VERIFY] Output text : \"%s\" (expected: \"17 * 23 is 391.\")\n", res.text.c_str());
        std::printf("[VERIFY] Token match : %s\n", exact ? "100% EXACT MATCH!" : "DIFF (check text/logits)");
        std::printf("[VERIFY] Speed       : prefill=%.2f tok/s (%.1f ms) | decode=%.2f tok/s (%.1f ms / %d tok)\n",
                    res.prompt_tokens * 1000.0 / std::max(res.prefill_ms, 0.001), res.prefill_ms,
                    res.gen_tokens * 1000.0 / std::max(res.decode_ms, 0.001), res.decode_ms, res.gen_tokens);
        if (do_profile) print_profile(engine);
    }

    if (do_verify_suite) {
        bool ok_suite = engine.run_parity_suite();
        if (!ok_suite) return 1;
    }

    if (do_probe_ee) {
        engine.probe_early_exit("Explain how a B-tree works in a database and write a short Python binary search function.", 48);
    }

    if (bench_prefill_n > 0) {
        std::printf("\n[BENCH PREFILL] Benchmarking %d-token realistic code/text prompt prefill (batched=%s)...\n",
                    bench_prefill_n, cfg.use_batched_prefill ? "ON (B=128 WMMA+PCIe)" : "OFF (B=1)");
        engine.reset_context();
        std::string sample_blk =
            "class VectorStore:\n"
            "    def __init__(self, dim: int, capacity: int = 65536):\n"
            "        self.dim = dim\n"
            "        self.capacity = capacity\n"
            "        self.entries = {}\n"
            "    def insert(self, key: str, vec: list[float]) -> None:\n"
            "        if len(vec) != self.dim: raise ValueError('dim mismatch')\n"
            "        self.entries[key] = vec\n"
            "    def query(self, vec: list[float], top_k: int = 8) -> list[tuple[str, float]]:\n"
            "        scores = [(k, sum(a*b for a, b in zip(v, vec))) for k, v in self.entries.items()]\n"
            "        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]\n\n";
        std::string big_text;
        while ((int)big_text.size() < bench_prefill_n * 5) big_text += sample_blk;
        std::string fmt = engine.tokenizer().format_chat({{"user", big_text}}, "none", true);
        auto ptoks = engine.tokenizer().encode(fmt, true);
        if ((int)ptoks.size() > bench_prefill_n) ptoks.resize(bench_prefill_n);
        if (engine.predictor_active()) engine.run_start_predictor(ptoks.data(), (int)ptoks.size());
        double pf_ms = 0.0;
        engine.reset_profile();
        engine.prefill(ptoks, 0, &pf_ms);
        std::printf("[BENCH PREFILL RESULT] %zu tokens in %.1f ms -> %.2f tok/s\n",
                    ptoks.size(), pf_ms, ptoks.size() * 1000.0 / std::max(pf_ms, 0.001));
        print_profile(engine);
    }

    if (bench_n > 0) {
        std::printf("\n[BENCH] Running %d-token decode benchmark (ctx=%d, start_pos=%d)...\n",
                    bench_n, cfg.max_ctx, bench_pos);
        engine.reset_context();

        // Populate context cleanly up to bench_pos if requested (fast synthetic KV fill for 32K-512K benchmark!)
        int cur_pos = 0;
        if (bench_pos > 0) {
            if (bench_pos + bench_n + 64 > cfg.max_ctx) {
                bench_pos = std::max(0, cfg.max_ctx - bench_n - 64);
            }
            if (bench_pos > 0) {
                std::printf("[BENCH] Populating 3-tier PagedKV cache up to pos=%d...\n", bench_pos);
                std::fflush(stdout);
                if (!engine.benchmark_populate_kv(bench_pos)) {
                    std::fprintf(stderr, "[BENCH] Failed populating KV cache!\n");
                    return 1;
                }
                cur_pos = bench_pos;
            }
        }

        // Warmup prompt + 8 tokens (engaging start-v3 & pred-v3 when enabled!)
        std::string bprompt = engine.tokenizer().format_chat(
            {{"user", "Write a detailed technical explanation of how Mixture-of-Experts looped transformers optimize memory bandwidth on consumer GPUs."}},
            "none", true);
        auto bids = engine.tokenizer().encode(bprompt, true);
        if (engine.predictor_active()) {
            engine.run_start_predictor(bids.data(), (int)bids.size());
        }
        int32_t tok = -1;
        for (size_t i = 0; i < bids.size(); ++i) {
            bool is_last = (i + 1 == bids.size());
            tok = engine.forward_token(bids[i], cur_pos++, is_last, false, false, is_last && engine.predictor_active());
        }
        if (engine.predictor_active()) {
            engine.run_main_predictor(0, true);
        }
        for (int w = 0; w < 8; ++w) {
            tok = engine.forward_token(tok, cur_pos++, true, false, true, false);
        }

        engine.reset_profile();
        auto t0 = std::chrono::steady_clock::now();
        std::string preview;
        for (int s = 0; s < bench_n; ++s) {
            bool do_pred = engine.predictor_active() && (((s + 8) % 16) == 0);
            tok = engine.forward_token(tok, cur_pos++, true, do_profile, true, do_pred);
            if (do_pred) engine.run_main_predictor(s + 8, false);
            if (s < 32) preview += engine.tokenizer().decode_token(tok);
            if (engine.tokenizer().is_eog(tok)) tok = 220; // keep benchmarking full bench_n tokens
        }
        double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        double tps = (bench_n * 1000.0) / elapsed_ms;

        std::printf("\n[BENCH RESULT] pos=%d | %d tokens in %.1f ms -> %.2f tok/s (%.2f ms/tok)\n",
                    cur_pos, bench_n, elapsed_ms, tps, elapsed_ms / bench_n);
        std::printf("[BENCH SAMPLE] \"%s...\"\n", preview.c_str());
        print_profile(engine);
    }

    if (!image_paths.empty() && prompt_str.empty()) {
        prompt_str = "Describe this image in detail.";
    }
    if (!prompt_str.empty()) {
        std::string p_eff = effort.empty() ? "none" : effort;
        ChatMessage user_msg;
        user_msg.role = "user";
        std::vector<EncodedImage> encoded_imgs;
        MultimodalEmbeddings mm_embd;

        if (!image_paths.empty()) {
            if (!engine.has_vision()) {
                std::fprintf(stderr, "[ERROR] --image specified, but vision encoder is not loaded (mmproj-F16.gguf missing or --no-vision set).\n");
                return 1;
            }
            encoded_imgs.reserve(image_paths.size());
            for (const auto& ipath : image_paths) {
                EncodedImage enc;
                if (!engine.encode_image_input(ipath, enc, /*allow_local_files=*/true)) {
                    std::fprintf(stderr, "[ERROR] Failed to encode image: %s\n", ipath.c_str());
                    return 1;
                }
                encoded_imgs.push_back(std::move(enc));
                ChatContentPart ipart;
                ipart.is_image = true;
                ipart.image_url = ipath;
                user_msg.parts.push_back(std::move(ipart));
            }
            ChatContentPart tpart;
            tpart.is_image = false;
            tpart.text = prompt_str;
            user_msg.parts.push_back(std::move(tpart));
        } else {
            user_msg.content = prompt_str;
        }

        std::string fmt = engine.tokenizer().format_chat({user_msg}, p_eff, true);
        auto raw_ids = engine.tokenizer().encode(fmt, true);
        std::vector<int32_t> ids;
        const MultimodalEmbeddings* mm_ptr = nullptr;
        if (!encoded_imgs.empty()) {
            InklingEngine::expand_multimodal_tokens(
                raw_ids, engine.tokenizer().image_placeholder_id(), encoded_imgs, ids, mm_embd);
            mm_ptr = &mm_embd;
        } else {
            ids = std::move(raw_ids);
        }

        std::printf("\n[Prompt] (%zu tokens%s) > ", ids.size(),
                    encoded_imgs.empty() ? "" : ", multimodal");
        std::fflush(stdout);
        engine.reset_profile();
        auto res = engine.generate(ids, sp, [&](int32_t id, const std::string& piece) {
            if (id == engine.tokenizer().thinking_id()) std::printf("[Thinking] ");
            else if (id == engine.tokenizer().text_id()) std::printf("\n[Answer] ");
            else std::printf("%s", piece.c_str());
            std::fflush(stdout);
            return true;
        }, do_profile, false, 0, nullptr, 0, 0, mm_ptr);
        std::printf("\n\n[Stats] prefill=%d tok (%.2f tok/s) | decode=%d tok (%.2f tok/s)",
                    res.prompt_tokens, res.prompt_tokens * 1000.0 / std::max(res.prefill_ms, 0.001),
                    res.gen_tokens, res.gen_tokens * 1000.0 / std::max(res.decode_ms, 0.001));
        if (res.spec_drafted > 0) {
            std::printf(" | spec=%d/%d accepted (%.1f%%)",
                        res.spec_accepted, res.spec_drafted, 100.0 * res.spec_accepted / res.spec_drafted);
        }
        std::printf("\n");
        if (do_profile) print_profile(engine);
    }

    if (do_serve) {
        srv_cfg.default_effort = effort.empty() ? "max" : effort;
        OpenAIServer srv(engine, srv_cfg);
        srv.run();
    }

    return 0;
}
