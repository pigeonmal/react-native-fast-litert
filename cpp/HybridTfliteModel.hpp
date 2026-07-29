#pragma once

#include "HybridTfliteModelSpec.hpp"

#include <litert/c/litert_compiled_model.h>
#include <litert/c/litert_environment.h>
#include <litert/c/litert_model.h>
#include <litert/c/litert_tensor_buffer.h>

#include <memory>
#include <mutex>
#include <vector>

namespace margelo::nitro::tflite {

class HybridTfliteModel final : public HybridTfliteModelSpec {
public:
  explicit HybridTfliteModel(LiteRtEnvironment environment, LiteRtModel model,
                             LiteRtCompiledModel compiledModel,
                             std::vector<LiteRtTensorBuffer> inputBuffers,
                             std::vector<LiteRtTensorBuffer> outputBuffers,
                             std::vector<Tensor> inputs, std::vector<Tensor> outputs,
                             std::shared_ptr<ArrayBuffer> modelData,
                             std::vector<TensorflowModelDelegate> delegates);
  ~HybridTfliteModel() override;

  std::vector<TensorflowModelDelegate> getDelegates() override;
  std::vector<Tensor> getInputs() override;
  std::vector<Tensor> getOutputs() override;

  std::vector<std::shared_ptr<ArrayBuffer>>
  runSync(const std::vector<std::shared_ptr<ArrayBuffer>>& input) override;
  std::shared_ptr<Promise<std::vector<std::shared_ptr<ArrayBuffer>>>>
  run(const std::vector<std::shared_ptr<ArrayBuffer>>& input) override;

protected:
  size_t getExternalMemorySize() noexcept override;

private:
  std::vector<std::shared_ptr<ArrayBuffer>>
  cloneInputBuffers(const std::vector<std::shared_ptr<ArrayBuffer>>& input) const;
  void copyInputBuffers(const std::vector<std::shared_ptr<ArrayBuffer>>& input);
  std::vector<std::shared_ptr<ArrayBuffer>> copyOutputBuffers();
  void rebuildForCpu();
  [[noreturn]] void throwRunError(LiteRtStatus status) const;
  std::vector<std::shared_ptr<ArrayBuffer>>
  runLocked(const std::vector<std::shared_ptr<ArrayBuffer>>& input);

private:
  LiteRtEnvironment _environment = nullptr;
  LiteRtModel _model = nullptr;
  LiteRtCompiledModel _compiledModel = nullptr;
  std::vector<LiteRtTensorBuffer> _inputBuffers;
  std::vector<LiteRtTensorBuffer> _outputBuffers;
  std::vector<Tensor> _inputs;
  std::vector<Tensor> _outputs;
  std::vector<std::shared_ptr<ArrayBuffer>> _outputArrayBuffers;
  std::vector<TensorflowModelDelegate> _delegates;
  std::shared_ptr<ArrayBuffer> _modelData;
  std::mutex _inferenceMutex;
};

} // namespace margelo::nitro::tflite
