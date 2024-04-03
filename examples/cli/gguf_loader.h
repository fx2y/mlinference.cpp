#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H
#include "model_loader.h"
#include <llama.h>
#include <common/common.h>
#include <iostream>

class GGUFLoader : public ModelLoader
{
private:
    llama_model *model;
    llama_context *ctx;

public:
    ~GGUFLoader()
    {
        unload();
    }

    bool load(const std::string &model_path) override
    {
        llama_model_params modelParams = llama_model_default_params();
        modelParams.n_gpu_layers = 99;
        model = llama_load_model_from_file(model_path.c_str(), modelParams);
        if (model == nullptr)
        {
            return false;
        }
        return initContext();
    }

    void unload() override
    {
        if (model != nullptr)
        {
            llama_free_model(model);
            model = nullptr;
        }
    }

    llama_context *getContext() {
        return ctx;
    }

    llama_model *getModel() {
        return model;
    }

    bool initContext() {
        llama_context_params ctxParams = llama_context_default_params();
        ctxParams.n_ctx = 2048;
        ctx = llama_new_context_with_model(model, ctxParams);
        if (ctx == nullptr)
        {
            llama_free_model(model);
            return false;
        }
        return true;
    }
};

#endif