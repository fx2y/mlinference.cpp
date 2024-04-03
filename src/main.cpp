#include <llama.h>
#include <iostream>
#include <common/common.h>

int main(int argc, char **argv)
{
    if (argc < 3 || argv[1][0] == '-')
    {
        printf("usage: %s MODEL_PATH PROMPT [--ids]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = argv[2];

    const bool printing_ids = argc > 3 && strcmp(argv[3], "--ids") == 0;

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    llama_model * model = llama_load_model_from_file(model_path, model_params);
}