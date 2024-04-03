#ifndef LLAMA_INFERER_H
#define LLAMA_INFERER_H
#include <memory>
#include "gguf_loader.h"
#include "model_inferer.h"
#include <llama.h>

class LlamaInferer : public ModelInferer
{
private:
    std::unique_ptr<GGUFLoader> modelLoader_;

public:
    LlamaInferer(const std::string &model_path)
    {
        modelLoader_ = std::make_unique<GGUFLoader>();
        if (!modelLoader_->load(model_path))
        {
            throw std::runtime_error("Failed to load model");
        }
    }

    void run(const void *inputData, void *outputData) override
    {
        auto inputTokens = static_cast<const std::vector<llama_token> *>(inputData);
        auto outputTokens = static_cast<std::vector<llama_token> *>(outputData);

        auto model = static_cast<llama_model *>(modelLoader_->getModel());
        if (model == nullptr)
        {
            throw std::runtime_error("Failed to load llama model");
        }

        auto ctx = static_cast<llama_context *>(modelLoader_->getContext());
        if (ctx == nullptr)
        {
            if (!modelLoader_->initContext())
            {
                throw std::runtime_error("Failed to create llama context");
            }
            ctx = static_cast<llama_context *>(modelLoader_->getContext());
        }

        // Create a batch for decoding
        llama_batch batch = llama_batch_init(512, 0, 1);

        // Evaluate the initial prompt
        for (size_t i = 0; i < inputTokens->size(); i++)
        {
            llama_batch_add(batch, (*inputTokens)[i], i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch) != 0)
        {
            llama_free(ctx);
            throw std::runtime_error("llama_decode() failed");
        }

        // Main generation loop
        int curLen = batch.n_tokens;
        while (true)
        {
            // Sample the next token
            auto nVocab = llama_n_vocab(model);
            auto *logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(nVocab);

            for (llama_token tokenId = 0; tokenId < nVocab; tokenId++)
            {
                candidates.emplace_back(llama_token_data{tokenId, logits[tokenId], 0.0f});
            }

            llama_token_data_array candidatesArray = {candidates.data(), candidates.size(), false};

            const llama_token newTokenId = llama_sample_token_greedy(ctx, &candidatesArray);

            if (newTokenId == llama_token_eos(model))
            {
                break;
            }

            // Append the new token to the tokens vector
            outputTokens->push_back(newTokenId);

            // Prepare the next batch
            llama_batch_clear(batch);
            llama_batch_add(batch, newTokenId, curLen, {0}, true);
            curLen++;

            // Evaluate the current batch
            if (llama_decode(ctx, batch))
            {
                llama_free(ctx);
                throw std::runtime_error("llama_decode() failed");
            }
        }

        llama_batch_free(batch);
        llama_free(ctx);
    }

    GGUFLoader *getModelLoader() {
        return modelLoader_.get();
    }
};
#endif