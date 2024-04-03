#include <iostream>
#include <string>
#include <vector>
#include <cxxopts.hpp>
#include <cpp-base64/base64.h>
#include <common/common.h>
#include <llama.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

std::string infer(const std::vector<uint8_t> &image_data)
{
    // Decode the image data
    int width, height, channels;
    unsigned char *img = stbi_load_from_memory(image_data.data(), image_data.size(), &width, &height, &channels, 0);
    if (img == nullptr)
    {
        return "Error decoding image";
    }

    // Process the image and return the result
    std::string result = "Image inference result: " + std::to_string(width) + "x" + std::to_string(height) + "x" + std::to_string(channels);

    stbi_image_free(img);
    return result;
}

std::string generateText(llama_context *ctx, llama_model *model, const std::string &prompt, int maxTokens)
{
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
        for (llama_token tokenId = 0; tokenId < nVocab; tokenId++)
        {
            candidates.emplace_back(llama_token_data{tokenId, logits[tokenId], 0.0f});
        }
        llama_token_data_array candidatesArray = {candidates.data(), candidates.size(), false};
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

    // Convert tokens to string
    std::string generatedText;
    for (auto id : tokens)
    {
        generatedText += llama_token_to_piece(ctx, id);
    }

    // Clean up
    llama_batch_free(batch);

    return generatedText;
}

int main(int argc, char **argv)
{
    gpt_params params;

    cxxopts::Options options("inference_tool", "A command line tool for text and image inference");
    options.add_options()("i,interactive", "Run in interactive mode")("t,text", "Text input for inference", cxxopts::value<std::string>())("b,base64", "Base64 encoded image input for inference", cxxopts::value<std::string>())("m,model", "Path to the model file", cxxopts::value<std::string>())("p,prompt", "Initial prompt", cxxopts::value<std::string>()->default_value(""))("h,help", "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }
    if (result.count("model"))
    {
        params.model = result["model"].as<std::string>();
    }
    else
    {
        std::cout << "Please provide the path to the model file using --model" << std::endl;
        return 1;
    }

    if (result.count("prompt"))
    {
        params.prompt = result["prompt"].as<std::string>();
    }

    if (params.prompt.empty())
    {
        params.prompt = "Hello my name is";
    }
    
    // Total length of the sequence including the prompt
    const int maxTokens = 32;

    // Initialize the LLaMA backend and model
    llama_backend_init();
    llama_model_params modelParams = llama_model_default_params();
    llama_model *model = llama_load_model_from_file(params.model.c_str(), modelParams);

    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // Create LLaMA context
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = 2048;
    llama_context *ctx = llama_new_context_with_model(model, ctxParams);
    if (ctx == NULL)
    {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    if (result.count("interactive"))
    {
        std::string input;
        while (true)
        {
            std::cout << "Enter text or base64 encoded image (or 'q' to quit): ";
            std::getline(std::cin, input);
            if (input == "q")
            {
                break;
            }

            if (input.find("data:image") != std::string::npos)
            {
                // Extract the base64 encoded image data
                auto pos = input.find(",");
                std::string base64_data = input.substr(pos + 1);
                std::string decoded_str = base64_decode(base64_data);
                std::vector<unsigned char> image_data;
                image_data.reserve(decoded_str.size());
                std::copy(decoded_str.begin(), decoded_str.end(), std::back_inserter(image_data));
                std::cout << infer(image_data) << std::endl;
            }
            else
            {
                std::cout << generateText(ctx, model, input, maxTokens) << std::endl;
            }
        }
    }
    else if (result.count("text"))
    {
        std::string text_input = result["text"].as<std::string>();
        std::cout << generateText(ctx, model, text_input, maxTokens) << std::endl;
    }
    else if (result.count("base64"))
    {
        std::string base64_input = result["base64"].as<std::string>();
        std::string decoded_str = base64_decode(base64_input);
        std::vector<unsigned char> image_data;
        image_data.reserve(decoded_str.size());
        std::copy(decoded_str.begin(), decoded_str.end(), std::back_inserter(image_data));
        std::cout << infer(image_data) << std::endl;
    }
    else
    {
        std::cout << "Please provide either --text or --base64 input" << std::endl;
        return 1;
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
