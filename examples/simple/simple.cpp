#include <common/common.h>
#include <llama.h>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    // Parse command line arguments
    std::string modelPath = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "Hello my name is";
    int maxTokens = 32;

    // Initialize the LLaMA backend and model
    llama_backend_init();
    llama_model_params modelParams = llama_model_default_params();
    llama_model *model = llama_load_model_from_file(modelPath.c_str(), modelParams);

    // Create LLaMA context
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = 2048;
    llama_context *ctx = llama_new_context_with_model(model, ctxParams);

    // Tokenize the prompt
    std::vector<llama_token> tokens = llama_tokenize(ctx, prompt, true);

    // Create a batch for decoding
    llama_batch batch = llama_batch_init(512, 0, 1);

    // Evaluate the initial prompt
    for (size_t i = 0; i < tokens.size(); i++)
    {
        llama_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    llama_decode(ctx, batch);

    // Main generation loop
    int curLen = batch.n_tokens;
    while (curLen <= maxTokens)
    {
        // Sample the next token
        auto logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        auto nVocab = llama_n_vocab(model);
        std::vector<llama_token_data> candidates;
        candidates.reserve(nVocab);
        for (llama_token tokenId = 0; tokenId < nVocab; tokenId++) {
            candidates.emplace_back(llama_token_data{ tokenId, logits[tokenId], 0.0f });
        }
        llama_token_data_array candidatesArray = { candidates.data(), candidates.size(), false };
        auto newTokenId = llama_sample_token_greedy(ctx, &candidatesArray);

        // Check for end of stream
        if (newTokenId == llama_token_eos(model) || curLen == maxTokens)
        {
            break;
        }

        // Append the new token to the tokens vector
        tokens.push_back(newTokenId);

        // Prepare the next batch
        llama_batch_clear(batch);
        llama_batch_add(batch, newTokenId, curLen, {0}, true);
        curLen++;

        // Evaluate the current batch
        llama_decode(ctx, batch);
    }

    // Print the generated text
    for (auto id : tokens)
    {
        printf("%s", llama_token_to_piece(ctx, id).c_str());
    }
    printf("\n");

    // Clean up
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}