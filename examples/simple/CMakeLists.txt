cmake_minimum_required(VERSION 3.10)
project(simple)

set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

find_package(Llama CONFIG REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB LLAMA_SOURCES "${LLAMA_INCLUDE_DIRS}/common/*.cpp")

add_executable(simple simple.cpp ${LLAMA_SOURCES})
target_link_libraries(simple PRIVATE llama ${CUDA_LIBRARIES})
target_include_directories(simple PRIVATE ${LLAMA_INCLUDE_DIRS})