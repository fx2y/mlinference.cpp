cmake_minimum_required(VERSION 3.10)
project(inference_tool)

set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

find_package(cxxopts CONFIG REQUIRED)
find_package(Llama CONFIG REQUIRED)

set(CPP_BASE64_SOURCES "${LLAMA_INCLUDE_DIRS}/cpp-base64/base64.cpp")
file(GLOB LLAMA_SOURCES "${LLAMA_INCLUDE_DIRS}/common/*.cpp")

add_executable(inference_tool main.cpp ${CPP_BASE64_SOURCES} ${LLAMA_SOURCES})

target_link_libraries(inference_tool PRIVATE cxxopts::cxxopts llama)

target_include_directories(inference_tool PRIVATE ${LLAMA_INCLUDE_DIRS})