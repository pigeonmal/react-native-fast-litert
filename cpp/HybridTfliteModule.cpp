#include "HybridTfliteModule.hpp"
#include "TfliteHelpers.hpp"

#include <litert/c/litert_compiled_model.h>
#include <litert/c/litert_environment.h>
#include <litert/c/litert_model.h>
#include <litert/c/litert_options.h>
#include <litert/c/litert_tensor_buffer.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

#if defined(__APPLE__)
#include <mach-o/dyld.h>

#include <filesystem>
#endif

namespace margelo::nitro::tflite {
namespace {

template <typename T, void (*Destroy)(T)> class LiteRtHandle final {
public:
  LiteRtHandle() = default;
  explicit LiteRtHandle(T value) : _value(value) {}
  ~LiteRtHandle() {
    reset();
  }

  LiteRtHandle(const LiteRtHandle&) = delete;
  LiteRtHandle& operator=(const LiteRtHandle&) = delete;

  LiteRtHandle(LiteRtHandle&& other) noexcept : _value(other.release()) {}
  LiteRtHandle& operator=(LiteRtHandle&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }

  T get() const {
    return _value;
  }

  T* out() {
    reset();
    return &_value;
  }

  T release() {
    return std::exchange(_value, nullptr);
  }

  void reset(T value = nullptr) {
    if (_value != nullptr) {
      Destroy(_value);
    }
    _value = value;
  }

private:
  T _value = nullptr;
};

using EnvironmentHandle = LiteRtHandle<LiteRtEnvironment, LiteRtDestroyEnvironment>;
using ModelHandle = LiteRtHandle<LiteRtModel, LiteRtDestroyModel>;
using OptionsHandle = LiteRtHandle<LiteRtOptions, LiteRtDestroyOptions>;
using CompiledModelHandle = LiteRtHandle<LiteRtCompiledModel, LiteRtDestroyCompiledModel>;

bool isGpuDelegate(TensorflowModelDelegate delegate) {
  return delegate == TensorflowModelDelegate::GPU || delegate == TensorflowModelDelegate::METAL ||
         delegate == TensorflowModelDelegate::ANDROID_GPU;
}

#if defined(__APPLE__)
std::string getMetalAcceleratorDirectory() {
  uint32_t executablePathSize = 0;
  _NSGetExecutablePath(nullptr, &executablePathSize);
  std::vector<char> executablePath(executablePathSize);
  if (_NSGetExecutablePath(executablePath.data(), &executablePathSize) != 0) {
    throw std::runtime_error("LiteRT: Failed to locate the iOS app bundle");
  }

  return (std::filesystem::path(executablePath.data()).parent_path() / "Frameworks" /
          "LiteRTMetalAccelerator.framework")
      .string();
}
#endif

void validateDelegates(const std::vector<TensorflowModelDelegate>& delegates) {
  for (TensorflowModelDelegate delegate : delegates) {
    if (delegate == TensorflowModelDelegate::CORE_ML) {
      throw std::runtime_error(
          "LiteRT: The Core ML delegate is not available in LiteRT 2.x CompiledModel. "
          "Use the cross-platform \"gpu\" accelerator instead.");
    }
    if (delegate == TensorflowModelDelegate::NNAPI) {
      throw std::runtime_error("LiteRT: NNAPI is not available in LiteRT 2.x CompiledModel. "
                               "Use the cross-platform \"gpu\" accelerator instead.");
    }
#if defined(ANDROID)
    if (delegate == TensorflowModelDelegate::METAL) {
      throw std::runtime_error("LiteRT: The \"metal\" GPU alias is only available on iOS. "
                               "Use \"gpu\" on cross-platform code.");
    }
#elif defined(__APPLE__)
    if (delegate == TensorflowModelDelegate::ANDROID_GPU) {
      throw std::runtime_error("LiteRT: The \"android-gpu\" alias is only available on Android. "
                               "Use \"gpu\" on cross-platform code.");
    }
#endif
  }
}

Tensor createTensor(LiteRtTensor tensor) {
  const char* name = nullptr;
  throwIfLiteRtError(LiteRtGetTensorName(tensor, &name), "read tensor name");

  LiteRtRankedTensorType tensorType{};
  throwIfLiteRtError(LiteRtGetRankedTensorType(tensor, &tensorType), "read tensor type");

  std::vector<double> shape;
  shape.reserve(tensorType.layout.rank);
  for (size_t dimension = 0; dimension < tensorType.layout.rank; ++dimension) {
    shape.push_back(static_cast<double>(tensorType.layout.dimensions[dimension]));
  }
  return Tensor(name == nullptr ? "" : name, getTensorDataType(tensorType.element_type),
                std::move(shape));
}

std::vector<Tensor> getSignatureTensors(LiteRtSignature signature, bool inputs) {
  LiteRtParamIndex count = 0;
  if (inputs) {
    throwIfLiteRtError(LiteRtGetNumSignatureInputs(signature, &count), "read signature inputs");
  } else {
    throwIfLiteRtError(LiteRtGetNumSignatureOutputs(signature, &count), "read signature outputs");
  }

  std::vector<Tensor> tensors;
  tensors.reserve(count);
  for (LiteRtParamIndex index = 0; index < count; ++index) {
    LiteRtTensor tensor = nullptr;
    if (inputs) {
      throwIfLiteRtError(LiteRtGetSignatureInputTensorByIndex(signature, index, &tensor),
                         "read signature input");
    } else {
      throwIfLiteRtError(LiteRtGetSignatureOutputTensorByIndex(signature, index, &tensor),
                         "read signature output");
    }
    tensors.push_back(createTensor(tensor));
  }
  return tensors;
}

} // namespace

std::shared_ptr<HybridTfliteModelSpec>
HybridTfliteModule::createModel(const std::shared_ptr<ArrayBuffer>& modelData,
                                const std::vector<TensorflowModelDelegate>& delegates) {
  validateDelegates(delegates);
  const bool gpuRequested = std::any_of(delegates.begin(), delegates.end(), isGpuDelegate);

  EnvironmentHandle environment;
#if defined(__APPLE__)
  const std::string runtimeLibraryDirectory =
      gpuRequested ? getMetalAcceleratorDirectory() : std::string();
  const LiteRtEnvOption environmentOption = {
      .tag = kLiteRtEnvOptionTagRuntimeLibraryDir,
      .value =
          {
              .type = kLiteRtAnyTypeString,
              .str_value = runtimeLibraryDirectory.c_str(),
          },
  };
  const int environmentOptionCount = gpuRequested ? 1 : 0;
  const LiteRtEnvOption* environmentOptions = gpuRequested ? &environmentOption : nullptr;
#else
  const int environmentOptionCount = 0;
  const LiteRtEnvOption* environmentOptions = nullptr;
#endif
  throwIfLiteRtError(
      LiteRtCreateEnvironment(environmentOptionCount, environmentOptions, environment.out()),
      "create environment");

  ModelHandle model;
  throwIfLiteRtError(LiteRtCreateModelFromBuffer(environment.get(), modelData->data(),
                                                 modelData->size(), model.out()),
                     "create model from data");

  LiteRtParamIndex signatureCount = 0;
  throwIfLiteRtError(LiteRtGetNumModelSignatures(model.get(), &signatureCount),
                     "read model signatures");
  if (signatureCount == 0) {
    throw std::runtime_error("LiteRT: Model does not define an inference signature");
  }

  LiteRtSignature signature = nullptr;
  throwIfLiteRtError(LiteRtGetModelSignature(model.get(), 0, &signature), "read model signature");
  std::vector<Tensor> inputs = getSignatureTensors(signature, true);
  std::vector<Tensor> outputs = getSignatureTensors(signature, false);

  const bool gpuSafe = !modelContainsInt16Tensors(model.get());
  bool useGpu = gpuRequested && gpuSafe;

  OptionsHandle options;
  throwIfLiteRtError(LiteRtCreateOptions(options.out()), "create compilation options");
  auto setAccelerators = [&options](bool gpu) {
    const LiteRtHwAcceleratorSet accelerators =
        gpu ? (kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu) : kLiteRtHwAcceleratorCpu;
    throwIfLiteRtError(LiteRtSetOptionsHardwareAccelerators(options.get(), accelerators),
                       "select hardware accelerators");
  };

  setAccelerators(useGpu);
  CompiledModelHandle compiledModel;
  LiteRtStatus compileStatus =
      LiteRtCreateCompiledModel(environment.get(), model.get(), options.get(), compiledModel.out());
  if (compileStatus != kLiteRtStatusOk && useGpu) {
    useGpu = false;
    setAccelerators(false);
    compileStatus = LiteRtCreateCompiledModel(environment.get(), model.get(), options.get(),
                                              compiledModel.out());
  }
  throwIfLiteRtError(compileStatus, "compile model");

  std::vector<LiteRtTensorBuffer> inputBuffers =
      createManagedTensorBuffers(environment.get(), compiledModel.get(), signature, true);
  std::vector<LiteRtTensorBuffer> outputBuffers;
  try {
    outputBuffers =
        createManagedTensorBuffers(environment.get(), compiledModel.get(), signature, false);
  } catch (...) {
    destroyTensorBuffers(inputBuffers);
    throw;
  }

  std::vector<TensorflowModelDelegate> resolvedDelegates =
      useGpu ? delegates : std::vector<TensorflowModelDelegate>{};
  return std::make_shared<HybridTfliteModel>(
      environment.release(), model.release(), compiledModel.release(), std::move(inputBuffers),
      std::move(outputBuffers), std::move(inputs), std::move(outputs), modelData,
      std::move(resolvedDelegates));
}

} // namespace margelo::nitro::tflite
