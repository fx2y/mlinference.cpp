#include <iostream>
#include <string>
#include <vector>
#include <cxxopts.hpp>
#include <cpp-base64/base64.h>
#include <common/common.h>
#include <llama.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "inference_engine.h"

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
    InferenceEngine engine = InferenceEngine(params.model);

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
                auto output = engine.run(input);
                std::cout << output << std::endl;
            }
        }
    }
    else if (result.count("text"))
    {
        std::string text_input = result["text"].as<std::string>();
        auto output = engine.run(text_input);
        std::cout << output << std::endl;
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
    llama_backend_free();

    return 0;
}
