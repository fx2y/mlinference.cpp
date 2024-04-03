#ifndef MODEL_INFERER_H
#define MODEL_INFERER_H
#include <string>

class ModelInferer
{
public:
    virtual void run(const void* inputData, void* outputData) = 0;
    virtual ~ModelInferer() {}
};
#endif