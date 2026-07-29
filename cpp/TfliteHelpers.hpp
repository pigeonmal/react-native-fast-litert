#pragma once

#include "TensorDataType.hpp"

#include <litert/c/litert_common.h>
#include <litert/c/litert_model_types.h>

#include <string>

namespace margelo::nitro::tflite {

void throwIfLiteRtError(LiteRtStatus status, const std::string& operation);
TensorDataType getTensorDataType(LiteRtElementType dataType);
bool modelContainsInt16Tensors(LiteRtModel model);

} // namespace margelo::nitro::tflite
