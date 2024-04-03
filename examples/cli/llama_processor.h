#include "inference_processor.h"
#include <string>
#include <llama.h>
#include <vector>
#include <memory>
#include "llama_inferer.h"

class LlamaProcessor : public InferenceProcessor
{
public:
    LlamaProcessor(GGUFLoader *modelLoader) : modelLoader_(modelLoader) {}
    ~LlamaProcessor() {}
    void *preProcess(const std::string &rawInput) override
    {
        // Pre-process the input data
        auto ctx = static_cast<llama_context *>(modelLoader_->getContext());
        auto output = std::make_unique<decltype(llama_tokenize(ctx, rawInput, true))>(llama_tokenize(ctx, rawInput, true));
        return output.release();
    }
    std::string postProcess(const void *modelOutput) override
    {
        // Post-process the model output
        auto ctx = static_cast<llama_context *>(modelLoader_->getContext());
        auto outputTokens = static_cast<const std::vector<llama_token> *>(modelOutput);
        std::string generatedText;
        for (size_t i = 0; i < outputTokens->size(); i++)
        {
            generatedText += llama_token_to_piece(ctx, (*outputTokens)[i]);
        }
        return generatedText;
    }

private:
    std::unique_ptr<GGUFLoader> modelLoader_;
};