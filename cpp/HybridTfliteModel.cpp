#include "HybridTfliteModel.hpp"
#include "TfliteHelpers.hpp"

#include <NitroModules/ArrayBuffer.hpp>
#include <NitroModules/Promise.hpp>

#include <litert/c/litert_options.h>

#include <cstring>
#include <stdexcept>

namespace margelo::nitro::tflite {
namespace {

class OptionsHandle final {
public:
  ~OptionsHandle() {
    if (_value != nullptr) {
      LiteRtDestroyOptions(_value);
    }
  }

  LiteRtOptions* out() {
    return &_value;
  }

  LiteRtOptions get() const {
    return _value;
  }

private:
  LiteRtOptions _value = nullptr;
};

} // namespace

HybridTfliteModel::HybridTfliteModel(LiteRtEnvironment environment, LiteRtModel model,
                                     LiteRtCompiledModel compiledModel,
                                     std::vector<LiteRtTensorBuffer> inputBuffers,
                                     std::vector<LiteRtTensorBuffer> outputBuffers,
                                     std::vector<Tensor> inputs, std::vector<Tensor> outputs,
                                     std::shared_ptr<ArrayBuffer> modelData,
                                     std::vector<TensorflowModelDelegate> delegates)
    : HybridObject(TAG), _environment(environment), _model(model), _compiledModel(compiledModel),
      _inputBuffers(std::move(inputBuffers)), _outputBuffers(std::move(outputBuffers)),
      _inputs(std::move(inputs)), _outputs(std::move(outputs)),
      _outputArrayBuffers(_outputBuffers.size()), _delegates(std::move(delegates)),
      _modelData(std::move(modelData)) {}

HybridTfliteModel::~HybridTfliteModel() {
  destroyTensorBuffers(_outputBuffers);
  destroyTensorBuffers(_inputBuffers);
  if (_compiledModel != nullptr) {
    LiteRtDestroyCompiledModel(_compiledModel);
  }
  if (_model != nullptr) {
    LiteRtDestroyModel(_model);
  }
  if (_environment != nullptr) {
    LiteRtDestroyEnvironment(_environment);
  }
}

std::vector<TensorflowModelDelegate> HybridTfliteModel::getDelegates() {
  std::lock_guard<std::mutex> lock(_inferenceMutex);
  return _delegates;
}

std::vector<Tensor> HybridTfliteModel::getInputs() {
  return _inputs;
}

std::vector<Tensor> HybridTfliteModel::getOutputs() {
  return _outputs;
}

size_t HybridTfliteModel::getExternalMemorySize() noexcept {
  size_t byteSize = _modelData == nullptr ? 0 : _modelData->size();
  for (LiteRtTensorBuffer buffer : _inputBuffers) {
    size_t bufferSize = 0;
    if (LiteRtGetTensorBufferSize(buffer, &bufferSize) == kLiteRtStatusOk) {
      byteSize += bufferSize;
    }
  }
  for (LiteRtTensorBuffer buffer : _outputBuffers) {
    size_t bufferSize = 0;
    if (LiteRtGetTensorBufferSize(buffer, &bufferSize) == kLiteRtStatusOk) {
      byteSize += bufferSize;
    }
  }
  return byteSize;
}

std::vector<std::shared_ptr<ArrayBuffer>>
HybridTfliteModel::cloneInputBuffers(const std::vector<std::shared_ptr<ArrayBuffer>>& input) const {
  std::vector<std::shared_ptr<ArrayBuffer>> copies;
  copies.reserve(input.size());
  for (const std::shared_ptr<ArrayBuffer>& buffer : input) {
    std::shared_ptr<ArrayBuffer> copy = ArrayBuffer::allocate(buffer->size());
    std::memcpy(copy->data(), buffer->data(), buffer->size());
    copies.push_back(std::move(copy));
  }
  return copies;
}

void HybridTfliteModel::copyInputBuffers(const std::vector<std::shared_ptr<ArrayBuffer>>& input) {
  if (input.size() != _inputBuffers.size()) {
    throw std::runtime_error("LiteRT: Input array size (" + std::to_string(input.size()) +
                             ") does not match input tensor count (" +
                             std::to_string(_inputBuffers.size()) + ")!");
  }

  for (size_t index = 0; index < input.size(); ++index) {
    size_t expectedSize = 0;
    throwIfLiteRtError(LiteRtGetTensorBufferPackedSize(_inputBuffers[index], &expectedSize),
                       "read input tensor size");
    if (input[index]->size() != expectedSize) {
      throw std::runtime_error("LiteRT: Input buffer " + std::to_string(index) + " has " +
                               std::to_string(input[index]->size()) + " bytes, expected " +
                               std::to_string(expectedSize) + " bytes");
    }

    void* destination = nullptr;
    throwIfLiteRtError(LiteRtLockTensorBuffer(_inputBuffers[index], &destination,
                                              kLiteRtTensorBufferLockModeWrite),
                       "lock input tensor");
    std::memcpy(destination, input[index]->data(), expectedSize);
    throwIfLiteRtError(LiteRtUnlockTensorBuffer(_inputBuffers[index]), "unlock input tensor");
  }
}

std::vector<std::shared_ptr<ArrayBuffer>> HybridTfliteModel::copyOutputBuffers() {
  std::vector<std::shared_ptr<ArrayBuffer>> results;
  results.reserve(_outputBuffers.size());

  for (size_t index = 0; index < _outputBuffers.size(); ++index) {
    size_t byteSize = 0;
    throwIfLiteRtError(LiteRtGetTensorBufferPackedSize(_outputBuffers[index], &byteSize),
                       "read output tensor size");
    if (_outputArrayBuffers[index] == nullptr || _outputArrayBuffers[index]->size() != byteSize) {
      _outputArrayBuffers[index] = ArrayBuffer::allocate(byteSize);
    }

    void* source = nullptr;
    throwIfLiteRtError(
        LiteRtLockTensorBuffer(_outputBuffers[index], &source, kLiteRtTensorBufferLockModeRead),
        "lock output tensor");
    std::memcpy(_outputArrayBuffers[index]->data(), source, byteSize);
    throwIfLiteRtError(LiteRtUnlockTensorBuffer(_outputBuffers[index]), "unlock output tensor");
    results.push_back(_outputArrayBuffers[index]);
  }
  return results;
}

void HybridTfliteModel::rebuildForCpu() {
  OptionsHandle options;
  throwIfLiteRtError(LiteRtCreateOptions(options.out()), "create CPU fallback options");
  throwIfLiteRtError(LiteRtSetOptionsHardwareAccelerators(options.get(), kLiteRtHwAcceleratorCpu),
                     "select CPU fallback");

  LiteRtCompiledModel cpuCompiledModel = nullptr;
  LiteRtStatus compileStatus =
      LiteRtCreateCompiledModel(_environment, _model, options.get(),
                                &cpuCompiledModel);
  if (compileStatus != kLiteRtStatusOk) {
    if (cpuCompiledModel != nullptr) {
      LiteRtDestroyCompiledModel(cpuCompiledModel);
    }
    throwIfLiteRtError(compileStatus, "compile CPU fallback");
  }

  std::vector<LiteRtTensorBuffer> cpuInputBuffers;
  std::vector<LiteRtTensorBuffer> cpuOutputBuffers;
  try {
    LiteRtSignature signature = nullptr;
    throwIfLiteRtError(LiteRtGetModelSignature(_model, 0, &signature),
                       "read CPU fallback signature");
    cpuInputBuffers = createManagedTensorBuffers(_environment, cpuCompiledModel, signature, true);
    cpuOutputBuffers = createManagedTensorBuffers(_environment, cpuCompiledModel, signature, false);
  } catch (...) {
    destroyTensorBuffers(cpuOutputBuffers);
    destroyTensorBuffers(cpuInputBuffers);
    LiteRtDestroyCompiledModel(cpuCompiledModel);
    throw;
  }

  destroyTensorBuffers(_outputBuffers);
  destroyTensorBuffers(_inputBuffers);
  LiteRtDestroyCompiledModel(_compiledModel);
  _compiledModel = cpuCompiledModel;
  _inputBuffers = std::move(cpuInputBuffers);
  _outputBuffers = std::move(cpuOutputBuffers);
  _outputArrayBuffers.assign(_outputBuffers.size(), nullptr);
  _delegates.clear();
}

[[noreturn]] void HybridTfliteModel::throwRunError(LiteRtStatus status) const {
  std::string message =
      "LiteRT: Failed to run model: " + std::string(LiteRtGetStatusString(status)) + " (" +
      std::to_string(static_cast<int>(status)) + ")";
  if (!_inputs.empty() && _inputs.front().dataType == TensorDataType::FLOAT16) {
    message += ". The model uses float16 input activations. LiteRT 2.1.6 CPU "
               "kernels do not support float16 Conv2D activations, and the GPU "
               "accelerator may reject a graph that requires CPU fallback. Export "
               "the model with float32 input/output tensors and float16-compressed "
               "weights, or use the float32 model.";
  }
  throw std::runtime_error(message);
}

std::vector<std::shared_ptr<ArrayBuffer>>
HybridTfliteModel::runLocked(const std::vector<std::shared_ptr<ArrayBuffer>>& input) {
  std::lock_guard<std::mutex> lock(_inferenceMutex);
  copyInputBuffers(input);
  LiteRtStatus status =
      LiteRtRunCompiledModel(_compiledModel, 0, _inputBuffers.size(), _inputBuffers.data(),
                             _outputBuffers.size(), _outputBuffers.data());
  if (status != kLiteRtStatusOk && !_delegates.empty()) {
    rebuildForCpu();
    copyInputBuffers(input);
    status = LiteRtRunCompiledModel(_compiledModel, 0, _inputBuffers.size(), _inputBuffers.data(),
                                    _outputBuffers.size(), _outputBuffers.data());
  }
  if (status != kLiteRtStatusOk) {
    throwRunError(status);
  }
  return copyOutputBuffers();
}

std::vector<std::shared_ptr<ArrayBuffer>>
HybridTfliteModel::runSync(const std::vector<std::shared_ptr<ArrayBuffer>>& input) {
  return runLocked(input);
}

std::shared_ptr<Promise<std::vector<std::shared_ptr<ArrayBuffer>>>>
HybridTfliteModel::run(const std::vector<std::shared_ptr<ArrayBuffer>>& input) {
  std::vector<std::shared_ptr<ArrayBuffer>> ownedInput = cloneInputBuffers(input);
  std::shared_ptr<HybridTfliteModel> sharedThis = shared_cast<HybridTfliteModel>();
  return Promise<std::vector<std::shared_ptr<ArrayBuffer>>>::async(
      [sharedThis, ownedInput = std::move(ownedInput)]() {
        return sharedThis->runLocked(ownedInput);
      });
}

} // namespace margelo::nitro::tflite
