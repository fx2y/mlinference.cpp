#include "inference_processor.h"
#include <string>
#include <llama.h>
#include <vector>
#include <memory>
#include "llama_inferer.h"

class LlamaProcessor : public InferenceProcessor
{
public:
    LlamaProcessor(llama_context *context) : context_(context) {}
    ~LlamaProcessor() {}
    void *preProcess(const std::string &rawInput) override
    {
        // Pre-process the input data
        auto output = std::make_unique<decltype(llama_tokenize(context_, rawInput, true))>(llama_tokenize(context_, rawInput, true));
        return output.release();
    }
    std::string postProcess(const void *modelOutput) override
    {
        // Post-process the model output
        auto outputTokens = static_cast<const std::vector<llama_token> *>(modelOutput);
        std::string generatedText;
        for (size_t i = 0; i < outputTokens->size(); i++)
        {
            generatedText += llama_token_to_piece(context_, (*outputTokens)[i]);
        }
        return generatedText;
    }

private:
    llama_context *context_;
};