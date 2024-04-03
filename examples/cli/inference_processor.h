class InferenceProcessor {
    public:
    virtual ~InferenceProcessor() {}
    virtual void* preProcess(const std::string &rawInput) = 0;
    virtual std::string postProcess(const void* modelOutput) = 0;
};