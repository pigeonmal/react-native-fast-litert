#include "TfliteHelpers.hpp"

#include <litert/c/litert_model.h>

#include <stdexcept>

namespace margelo::nitro::tflite {
namespace {

bool tensorIsInt16(LiteRtTensor tensor) {
  LiteRtTensorTypeId typeId;
  throwIfLiteRtError(LiteRtGetTensorTypeId(tensor, &typeId), "read tensor type");

  if (typeId == kLiteRtRankedTensorType) {
    LiteRtRankedTensorType type{};
    throwIfLiteRtError(LiteRtGetRankedTensorType(tensor, &type), "read ranked tensor type");
    return type.element_type == kLiteRtElementTypeInt16;
  }

  LiteRtUnrankedTensorType type{};
  throwIfLiteRtError(LiteRtGetUnrankedTensorType(tensor, &type), "read unranked tensor type");
  return type.element_type == kLiteRtElementTypeInt16;
}

} // namespace

void throwIfLiteRtError(LiteRtStatus status, const std::string& operation) {
  if (status == kLiteRtStatusOk) {
    return;
  }
  throw std::runtime_error("LiteRT: Failed to " + operation + ": " +
                           std::string(LiteRtGetStatusString(status)) + " (" +
                           std::to_string(static_cast<int>(status)) + ")");
}

TensorDataType getTensorDataType(LiteRtElementType dataType) {
  switch (dataType) {
    case kLiteRtElementTypeFloat16:
      return TensorDataType::FLOAT16;
    case kLiteRtElementTypeFloat32:
      return TensorDataType::FLOAT32;
    case kLiteRtElementTypeFloat64:
      return TensorDataType::FLOAT64;
    case kLiteRtElementTypeBFloat16:
      return TensorDataType::BFLOAT16;
    case kLiteRtElementTypeInt4:
      return TensorDataType::INT4;
    case kLiteRtElementTypeInt8:
      return TensorDataType::INT8;
    case kLiteRtElementTypeInt16:
      return TensorDataType::INT16;
    case kLiteRtElementTypeInt32:
      return TensorDataType::INT32;
    case kLiteRtElementTypeInt64:
      return TensorDataType::INT64;
    case kLiteRtElementTypeUInt8:
      return TensorDataType::UINT8;
    case kLiteRtElementTypeUInt16:
      return TensorDataType::UINT16;
    case kLiteRtElementTypeUInt32:
      return TensorDataType::UINT32;
    case kLiteRtElementTypeUInt64:
      return TensorDataType::UINT64;
    case kLiteRtElementTypeNone:
      return TensorDataType::NONE;
    case kLiteRtElementTypeTfString:
      return TensorDataType::STRING;
    case kLiteRtElementTypeBool:
      return TensorDataType::BOOL;
    case kLiteRtElementTypeComplex64:
      return TensorDataType::COMPLEX64;
    case kLiteRtElementTypeComplex128:
      return TensorDataType::COMPLEX128;
    case kLiteRtElementTypeTfResource:
      return TensorDataType::RESOURCE;
    case kLiteRtElementTypeTfVariant:
      return TensorDataType::VARIANT;
    case kLiteRtElementTypeInt2:
    case kLiteRtElementTypeUInt4:
    case kLiteRtElementTypeFloat8E4M3FN:
    case kLiteRtElementTypeFloat8E5M2:
      break;
  }
  throw std::runtime_error("LiteRT: Unsupported tensor element type " +
                           std::to_string(static_cast<int>(dataType)));
}

bool modelContainsInt16Tensors(LiteRtModel model) {
  LiteRtParamIndex subgraphCount = 0;
  throwIfLiteRtError(LiteRtGetNumModelSubgraphs(model, &subgraphCount), "read model subgraphs");

  for (LiteRtParamIndex subgraphIndex = 0; subgraphIndex < subgraphCount; ++subgraphIndex) {
    LiteRtSubgraph subgraph = nullptr;
    throwIfLiteRtError(LiteRtGetModelSubgraph(model, subgraphIndex, &subgraph),
                       "read model subgraph");

    LiteRtParamIndex operationCount = 0;
    throwIfLiteRtError(LiteRtGetNumSubgraphOps(subgraph, &operationCount),
                       "read subgraph operations");
    for (LiteRtParamIndex operationIndex = 0; operationIndex < operationCount; ++operationIndex) {
      LiteRtOp operation = nullptr;
      throwIfLiteRtError(LiteRtGetSubgraphOp(subgraph, operationIndex, &operation),
                         "read subgraph operation");

      LiteRtParamIndex inputCount = 0;
      throwIfLiteRtError(LiteRtGetNumOpInputs(operation, &inputCount), "read operation inputs");
      for (LiteRtParamIndex inputIndex = 0; inputIndex < inputCount; ++inputIndex) {
        LiteRtTensor tensor = nullptr;
        throwIfLiteRtError(LiteRtGetOpInput(operation, inputIndex, &tensor),
                           "read operation input");
        if (tensorIsInt16(tensor)) {
          return true;
        }
      }

      LiteRtParamIndex outputCount = 0;
      throwIfLiteRtError(LiteRtGetNumOpOutputs(operation, &outputCount), "read operation outputs");
      for (LiteRtParamIndex outputIndex = 0; outputIndex < outputCount; ++outputIndex) {
        LiteRtTensor tensor = nullptr;
        throwIfLiteRtError(LiteRtGetOpOutput(operation, outputIndex, &tensor),
                           "read operation output");
        if (tensorIsInt16(tensor)) {
          return true;
        }
      }
    }
  }
  return false;
}

std::vector<LiteRtTensorBuffer> createManagedTensorBuffers(LiteRtEnvironment environment,
                                                           LiteRtCompiledModel compiledModel,
                                                           LiteRtSignature signature, bool inputs) {
  LiteRtParamIndex count = 0;
  if (inputs) {
    throwIfLiteRtError(LiteRtGetNumSignatureInputs(signature, &count), "read signature inputs");
  } else {
    throwIfLiteRtError(LiteRtGetNumSignatureOutputs(signature, &count), "read signature outputs");
  }

  std::vector<LiteRtTensorBuffer> buffers;
  buffers.reserve(count);
  try {
    for (LiteRtParamIndex index = 0; index < count; ++index) {
      LiteRtTensor tensor = nullptr;
      if (inputs) {
        throwIfLiteRtError(LiteRtGetSignatureInputTensorByIndex(signature, index, &tensor),
                           "read signature input");
      } else {
        throwIfLiteRtError(LiteRtGetSignatureOutputTensorByIndex(signature, index, &tensor),
                           "read signature output");
      }

      LiteRtRankedTensorType tensorType{};
      throwIfLiteRtError(LiteRtGetRankedTensorType(tensor, &tensorType), "read tensor type");

      LiteRtTensorBufferRequirements requirements = nullptr;
      if (inputs) {
        throwIfLiteRtError(
            LiteRtGetCompiledModelInputBufferRequirements(compiledModel, 0, index, &requirements),
            "read input buffer requirements");
      } else {
        throwIfLiteRtError(
            LiteRtGetCompiledModelOutputBufferRequirements(compiledModel, 0, index, &requirements),
            "read output buffer requirements");
      }

      LiteRtTensorBuffer buffer = nullptr;
      throwIfLiteRtError(LiteRtCreateManagedTensorBufferFromRequirements(environment, &tensorType,
                                                                         requirements, &buffer),
                         "create tensor buffer");
      buffers.push_back(buffer);
    }
    return buffers;
  } catch (...) {
    destroyTensorBuffers(buffers);
    throw;
  }
}

void destroyTensorBuffers(std::vector<LiteRtTensorBuffer>& buffers) noexcept {
  for (LiteRtTensorBuffer buffer : buffers) {
    LiteRtDestroyTensorBuffer(buffer);
  }
  buffers.clear();
}

} // namespace margelo::nitro::tflite
