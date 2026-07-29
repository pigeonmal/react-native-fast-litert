#pragma once

#include "TensorDataType.hpp"

#include <litert/c/litert_compiled_model.h>
#include <litert/c/litert_environment.h>
#include <litert/c/litert_model.h>
#include <litert/c/litert_tensor_buffer.h>

#include <string>
#include <vector>

namespace margelo::nitro::tflite {

void throwIfLiteRtError(LiteRtStatus status, const std::string& operation);
TensorDataType getTensorDataType(LiteRtElementType dataType);
bool modelContainsInt16Tensors(LiteRtModel model);
std::vector<LiteRtTensorBuffer> createManagedTensorBuffers(LiteRtEnvironment environment,
                                                           LiteRtCompiledModel compiledModel,
                                                           LiteRtSignature signature, bool inputs);
void destroyTensorBuffers(std::vector<LiteRtTensorBuffer>& buffers) noexcept;

} // namespace margelo::nitro::tflite
