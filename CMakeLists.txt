cmake_minimum_required(VERSION 3.10)
project(my_project)

set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

find_package(Llama CONFIG REQUIRED)
find_package(CUDA REQUIRED)

add_executable(my_project src/main.cpp)
target_link_libraries(my_project PRIVATE llama ${CUDA_LIBRARIES})
target_include_directories(my_project PRIVATE ${LLAMA_INCLUDE_DIRS})