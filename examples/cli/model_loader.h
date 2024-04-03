#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H
#include <string>

class ModelLoader
{
public:
    virtual ~ModelLoader() {}
    virtual bool load(const std::string &model_path) = 0;
    virtual void unload() = 0;
};
#endif