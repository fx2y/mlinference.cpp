#include <memory>
#include <vector>
#include "llama_processor.h"
#include "llama_inferer.h"
#include "llama.h"

class InferenceEngine
{
private:
    std::unique_ptr<LlamaInferer> modelInferer_;
    std::unique_ptr<LlamaProcessor> inferenceProcessor_;

public:
    InferenceEngine(const std::string &model_path)
    {
        modelInferer_ = std::make_unique<LlamaInferer>(model_path);
        inferenceProcessor_ = std::make_unique<LlamaProcessor>(modelInferer_->getModelLoader()->getContext());
    }
    std::string run(const std::string &input)
    {
        // Pre-process the input data
        auto preProcessedInput = inferenceProcessor_->preProcess(input);

        // Run inference using the processed input
        std::vector<llama_token> *outputTokens = new std::vector<llama_token>();
        modelInferer_->run(preProcessedInput, outputTokens);

        // Post-process the model output
        return inferenceProcessor_->postProcess(outputTokens);
    }
    ~InferenceEngine() {}
};