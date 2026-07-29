#include <litert/c/litert_compiled_model.h>
#include <litert/c/litert_environment.h>
#include <litert/c/litert_model.h>
#include <litert/c/litert_options.h>
#include <litert/c/litert_tensor_buffer.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(LiteRtStatus status, const char* operation) {
  if (status != kLiteRtStatusOk) {
    throw std::runtime_error(std::string(operation) + ": " + LiteRtGetStatusString(status));
  }
}

std::vector<uint8_t> readFile(const char* path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error(std::string("Could not open ") + path);
  }
  const auto size = stream.tellg();
  stream.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  return bytes;
}

std::vector<LiteRtTensorBuffer> createBuffers(LiteRtEnvironment environment,
                                               LiteRtCompiledModel compiledModel,
                                               LiteRtSignature signature, bool inputs) {
  LiteRtParamIndex count = 0;
  check(inputs ? LiteRtGetNumSignatureInputs(signature, &count)
               : LiteRtGetNumSignatureOutputs(signature, &count),
        "Get tensor count");

  std::vector<LiteRtTensorBuffer> buffers;
  buffers.reserve(count);
  for (LiteRtParamIndex index = 0; index < count; ++index) {
    LiteRtTensor tensor = nullptr;
    check(inputs ? LiteRtGetSignatureInputTensorByIndex(signature, index, &tensor)
                 : LiteRtGetSignatureOutputTensorByIndex(signature, index, &tensor),
          "Get tensor");
    LiteRtRankedTensorType type{};
    check(LiteRtGetRankedTensorType(tensor, &type), "Get tensor type");
    LiteRtTensorBufferRequirements requirements = nullptr;
    check(inputs ? LiteRtGetCompiledModelInputBufferRequirements(compiledModel, 0, index,
                                                                 &requirements)
                 : LiteRtGetCompiledModelOutputBufferRequirements(compiledModel, 0, index,
                                                                  &requirements),
          "Get buffer requirements");
    LiteRtTensorBuffer buffer = nullptr;
    check(LiteRtCreateManagedTensorBufferFromRequirements(environment, &type, requirements,
                                                           &buffer),
          "Create tensor buffer");
    buffers.push_back(buffer);
  }
  return buffers;
}

void clearBuffer(LiteRtTensorBuffer buffer) {
  size_t byteSize = 0;
  check(LiteRtGetTensorBufferPackedSize(buffer, &byteSize), "Get buffer size");
  void* data = nullptr;
  check(LiteRtLockTensorBuffer(buffer, &data, kLiteRtTensorBufferLockModeWrite),
        "Lock tensor buffer");
  std::memset(data, 0, byteSize);
  check(LiteRtUnlockTensorBuffer(buffer), "Unlock tensor buffer");
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: litert-model-smoke-test MODEL [cpu|gpu]\n";
    return 2;
  }

  LiteRtEnvironment environment = nullptr;
  LiteRtModel model = nullptr;
  LiteRtOptions options = nullptr;
  LiteRtCompiledModel compiledModel = nullptr;
  std::vector<LiteRtTensorBuffer> inputs;
  std::vector<LiteRtTensorBuffer> outputs;

  try {
    std::vector<uint8_t> modelBytes = readFile(argv[1]);
    const char* runtimeLibraryDirectory = std::getenv("LITERT_RUNTIME_LIBRARY_DIR");
    const LiteRtEnvOption environmentOption = {
        .tag = kLiteRtEnvOptionTagRuntimeLibraryDir,
        .value =
            {
                .type = kLiteRtAnyTypeString,
                .str_value = runtimeLibraryDirectory,
            },
    };
    check(LiteRtCreateEnvironment(runtimeLibraryDirectory == nullptr ? 0 : 1,
                                  runtimeLibraryDirectory == nullptr ? nullptr
                                                                     : &environmentOption,
                                  &environment),
          "Create environment");
    check(LiteRtCreateModelFromBuffer(environment, modelBytes.data(), modelBytes.size(), &model),
          "Create model");
    check(LiteRtCreateOptions(&options), "Create options");
    const bool useGpu = argc == 3 && std::string(argv[2]) == "gpu";
    check(LiteRtSetOptionsHardwareAccelerators(
              options, useGpu ? kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu
                              : kLiteRtHwAcceleratorCpu),
          "Set accelerators");
    check(LiteRtCreateCompiledModel(environment, model, options, &compiledModel),
          "Compile model");

    LiteRtSignature signature = nullptr;
    check(LiteRtGetModelSignature(model, 0, &signature), "Get signature");
    inputs = createBuffers(environment, compiledModel, signature, true);
    outputs = createBuffers(environment, compiledModel, signature, false);
    for (LiteRtTensorBuffer input : inputs) {
      clearBuffer(input);
    }
    check(LiteRtRunCompiledModel(compiledModel, 0, inputs.size(), inputs.data(), outputs.size(),
                                 outputs.data()),
          "Run model");

    std::cout << argv[1] << " ran successfully on " << (useGpu ? "GPU/CPU" : "CPU")
              << " with " << outputs.size() << " output tensor(s)\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    for (LiteRtTensorBuffer buffer : outputs) LiteRtDestroyTensorBuffer(buffer);
    for (LiteRtTensorBuffer buffer : inputs) LiteRtDestroyTensorBuffer(buffer);
    if (compiledModel != nullptr) LiteRtDestroyCompiledModel(compiledModel);
    if (options != nullptr) LiteRtDestroyOptions(options);
    if (model != nullptr) LiteRtDestroyModel(model);
    if (environment != nullptr) LiteRtDestroyEnvironment(environment);
    return 1;
  }

  for (LiteRtTensorBuffer buffer : outputs) LiteRtDestroyTensorBuffer(buffer);
  for (LiteRtTensorBuffer buffer : inputs) LiteRtDestroyTensorBuffer(buffer);
  LiteRtDestroyCompiledModel(compiledModel);
  LiteRtDestroyOptions(options);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
  return 0;
}
