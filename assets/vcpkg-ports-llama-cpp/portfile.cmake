vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO ggerganov/llama.cpp
    REF ${VERSION}
    SHA512 bf55c09a46b1544efbd3bc3cedbbe69fa3bdb81742175444a3b58c9f78a9496e8fc59b7e9275366a4d293679fcccea500a3a31111b2bb8131749f5b15fb3c061
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS
        -DLLAMA_METAL:BOOL=ON
        -DLLAMA_BUILD_TESTS:BOOL=ON
        -DLLAMA_BUILD_EXAMPLES:BOOL=OFF
        -DLLAMA_BUILD_SERVER:BOOL=OFF
        -DLLAMA_ACCELERATE:BOOL=OFF
    MAYBE_UNUSED_VARIABLES
        LLAMA_BUILD_SERVER
)
vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/Llama PACKAGE_NAME llama)

file(COPY ${SOURCE_PATH}/common DESTINATION ${CURRENT_PACKAGES_DIR}/include)

vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/llama/LlamaConfig.cmake" "/../../../" "/../../")
vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/llama/LlamaConfig.cmake" "${CURRENT_INSTALLED_DIR}" "\${PACKAGE_PREFIX_DIR}")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE" )
