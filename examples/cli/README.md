## Quickstart

1. Install VCPKG for C++ dependency manager. Refer to https://github.com/microsoft/vcpkg. Assume that it's installed on `~/opt`
```shell
mkdir -p ~/opt/vcpkg-downloads && cd ~/opt
git clone https://github.com/microsoft/vcpkg
./vcpkg/bootstrap-vcpkg.sh
# Set VCPKG_ROOT and VCPKG_DOWNLOADS accordingly
export VCPKG_ROOT=~/opt/vcpkg
export VCPKG_DOWNLOADS=~/opt/vcpkg-downloads
```

2. Copy custom ports for llama.cpp from repo into VCPKG_ROOT
```shell
cp -r ../../assets/vcpkg-ports-llama-cpp $VCPKG_ROOT/ports/llama-cpp
```

3. Install required dependencies
```shell
$VCPKG_ROOT/vcpkg install cxxopts cpp-base64 stb llama-cpp
```

4. Build and run the program
```shell
mkdir -p build && cd build
cmake ..
cmake --build .
./inference-tool -m <some-model.gguf> -t "Hi my name is"
# or ./inference_tool -m <some-model.gguf> -i for interactive mode
# or ./inference_tool -m <some-model.gguf> -b <base64 image> for vision transformer model
```
