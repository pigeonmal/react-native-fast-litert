project(NitroTflite)
cmake_minimum_required(VERSION 3.22.1)

set (PACKAGE_NAME NitroTflite)
set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_CXX_STANDARD 20)

# Find the LiteRT 2.x runtime extracted from the Maven AAR.
find_library(
  LITERT
  LiteRt
  PATHS "./src/main/cpp/lib/litert/jni/${ANDROID_ABI}"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

find_library(
  LITERT_GPU
  LiteRtClGlAccelerator
  PATHS "./src/main/cpp/lib/litert/jni/${ANDROID_ABI}"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(NOT LITERT OR NOT LITERT_GPU)
  message(FATAL_ERROR "LiteRT 2.1.6 runtime or GPU accelerator was not extracted for ${ANDROID_ABI}")
endif()

# -DANDROID flag
string(APPEND CMAKE_CXX_FLAGS " -DANDROID")

# Define C++ library and add all sources
add_library(${PACKAGE_NAME} SHARED
        src/main/cpp/cpp-adapter.cpp
        ../cpp/HybridTfliteModel.cpp
        ../cpp/HybridTfliteModule.cpp
        ../cpp/TfliteHelpers.cpp
)

# Add Nitrogen specs :)
include(${CMAKE_SOURCE_DIR}/../nitrogen/generated/android/NitroTflite+autolinking.cmake)

# Set up local includes
include_directories(
        "src/main/cpp"
        "../cpp"
        "../third_party/litert/include"
)

find_library(LOG_LIB log)
find_library(DL_LIB dl)

# Link all libraries together
target_link_libraries(
        ${PACKAGE_NAME}
        ${LOG_LIB}
        ${DL_LIB}
        android
        ${LITERT}
        ${LITERT_GPU}
)
