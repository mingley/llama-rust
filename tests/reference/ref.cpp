// Reference dump from llama.cpp: token ids, prefill logits, greedy continuation.
// Built against the locally built libllama so the comparison uses one GGUF and
// one host.
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt> <n_predict>\n", argv[0]);
        return 1;
    }
    const char * path   = argv[1];
    std::string  prompt = argv[2];
    const int    npred  = atoi(argv[3]);

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 512;
    cp.n_batch = 512;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    // Tokenize exactly as the model metadata dictates (add_special honours
    // add_bos_token; parse_special handles <|im_start|> style tokens).
    int need = -llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                               nullptr, 0, true, true);
    std::vector<llama_token> toks(need);
    int ntok = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                              toks.data(), need, true, true);
    toks.resize(ntok);

    printf("N_VOCAB %d\n", n_vocab);
    printf("TOKENS %d:", ntok);
    for (auto t : toks) printf(" %d", t);
    printf("\n");
    printf("PIECES:");
    for (auto t : toks) {
        char buf[256];
        int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        printf(" [%.*s]", n > 0 ? n : 0, buf);
    }
    printf("\n");

    // Prefill the whole prompt in one batch, then read last-position logits.
    llama_batch batch = llama_batch_get_one(toks.data(), (int) toks.size());
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    float * logits = llama_get_logits_ith(ctx, (int) toks.size() - 1);

    // Full-vector summary plus the top-20 ranking, which is what actually
    // determines greedy output.
    double sum = 0.0, sumsq = 0.0;
    float mn = logits[0], mx = logits[0];
    for (int i = 0; i < n_vocab; i++) {
        sum += logits[i]; sumsq += (double) logits[i] * logits[i];
        mn = std::min(mn, logits[i]); mx = std::max(mx, logits[i]);
    }
    printf("LOGIT_STATS min %.6f max %.6f mean %.6f rms %.6f\n",
           mn, mx, sum / n_vocab, sqrt(sumsq / n_vocab));

    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    printf("TOP20:");
    for (int i = 0; i < 20; i++) printf(" %d:%.6f", idx[i], logits[idx[i]]);
    printf("\n");

    // Greedy continuation, argmax each step.
    std::string out;
    std::vector<llama_token> gen;
    int n_past = (int) toks.size();
    llama_token cur = idx[0];
    for (int s = 0; s < npred; s++) {
        gen.push_back(cur);
        char buf[256];
        int n = llama_token_to_piece(vocab, cur, buf, sizeof(buf), 0, true);
        if (n > 0) out.append(buf, n);
        llama_batch b1 = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, b1) != 0) break;
        float * lg = llama_get_logits_ith(ctx, 0);
        int best = 0;
        for (int i = 1; i < n_vocab; i++) if (lg[i] > lg[best]) best = i;
        cur = best;
        n_past++;
    }
    printf("GREEDY_IDS:");
    for (auto t : gen) printf(" %d", t);
    printf("\n");
    printf("GREEDY_TEXT [%s]\n", out.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
