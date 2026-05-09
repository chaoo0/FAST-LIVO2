#include "pose_compensator.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <sstream>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#ifdef HAVE_ONNXRUNTIME
namespace
{
using TensorElementType = ONNXTensorElementDataType;
constexpr size_t kExpectedInputRank = 2;
constexpr size_t kExpectedOutputRank = 1;
constexpr int64_t kExpectedFeatureDim = 18;
constexpr int64_t kExpectedOutputDim = 6;

bool matchesInputDim(const int64_t model_dim, const int64_t actual_dim)
{
  return model_dim <= 0 || model_dim == actual_dim;
}

bool isSupportedElementType(const TensorElementType type)
{
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

std::string tensorElementTypeToString(const TensorElementType type)
{
  switch (type)
  {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return "float";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return "double";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    return "undefined";
  default:
    return std::to_string(static_cast<int>(type));
  }
}

std::string shapeToString(const std::vector<int64_t> &shape)
{
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i)
  {
    if (i > 0)
    {
      oss << ", ";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

int findInputIndexByName(const Ort::Session &session, const std::string &target_name)
{
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t input_count = session.GetInputCount();
  for (size_t i = 0; i < input_count; ++i)
  {
    auto input_name = session.GetInputNameAllocated(i, allocator);
    if (input_name && target_name == input_name.get())
    {
      return static_cast<int>(i);
    }
  }
  return -1;
}

int findOutputIndexByName(const Ort::Session &session, const std::string &target_name)
{
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t output_count = session.GetOutputCount();
  for (size_t i = 0; i < output_count; ++i)
  {
    auto output_name = session.GetOutputNameAllocated(i, allocator);
    if (output_name && target_name == output_name.get())
    {
      return static_cast<int>(i);
    }
  }
  return -1;
}

bool validateModelInputShape(const std::vector<int64_t> &model_shape, const int64_t sequence_length,
                             const int64_t feature_dim)
{
  if (model_shape.size() != kExpectedInputRank)
  {
    return false;
  }
  return matchesInputDim(model_shape[0], sequence_length) && matchesInputDim(model_shape[1], feature_dim);
}

template <typename TensorShapeInfoT>
bool readExpectedRankShape(const TensorShapeInfoT &tensor_info, const size_t expected_rank,
                           const std::string &tensor_label, std::vector<int64_t> &shape, std::string &error)
{
  const size_t actual_rank = tensor_info.GetDimensionsCount();
  if (actual_rank != expected_rank)
  {
    std::ostringstream oss;
    oss << "expected " << tensor_label << " rank " << expected_rank << ", got " << actual_rank;
    error = oss.str();
    return false;
  }

  shape.assign(expected_rank, 0);
  if (!shape.empty())
  {
    Ort::ThrowOnError(Ort::GetApi().GetDimensions(tensor_info, shape.data(), shape.size()));
  }
  error = "none";
  return true;
}

std::string inputShapeMismatchMessage(const std::vector<int64_t> &model_shape, const int64_t sequence_length,
                                      const int64_t feature_dim)
{
  std::ostringstream oss;
  oss << "model input shape " << shapeToString(model_shape)
      << " is incompatible with runtime shape [" << sequence_length << ", " << feature_dim << "]";
  return oss.str();
}
} // namespace
#endif

#ifdef HAVE_ONNXRUNTIME
struct PoseCompensator::OnnxInferenceBackend::Impl
{
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::SessionOptions> session_options;
  std::unique_ptr<Ort::Session> session;
  TensorElementType input_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  TensorElementType output_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
};
#else
struct PoseCompensator::OnnxInferenceBackend::Impl
{
};
#endif

PoseCompensator::PoseCompensator() : backend_(std::make_unique<DummyInferenceBackend>()) {}

PoseCompensator::PoseCorrection PoseCompensator::DummyInferenceBackend::infer(const FlatFeatureInput &input) const
{
  (void)input;
  last_inference_success_ = false;
  last_inference_status_ = "dummy_zero_output";
  return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

std::string PoseCompensator::DummyInferenceBackend::name() const
{
  return "dummy";
}

std::string PoseCompensator::DummyInferenceBackend::statusMessage() const
{
  return "dummy_zero_output";
}

bool PoseCompensator::DummyInferenceBackend::lastInferenceSuccess() const
{
  return last_inference_success_;
}

std::string PoseCompensator::DummyInferenceBackend::lastInferenceStatus() const
{
  return last_inference_status_;
}

PoseCompensator::OnnxPlaceholderBackend::OnnxPlaceholderBackend(std::string model_path)
    : model_path_(std::move(model_path)) {}

PoseCompensator::PoseCorrection PoseCompensator::OnnxPlaceholderBackend::infer(const FlatFeatureInput &input) const
{
  const bool shape_valid = input.sequence_length > 0 && input.feature_dim > 0;
  const bool flat_len_valid = input.flat_input_length == input.sequence_length * input.feature_dim;
  const bool data_size_valid = input.data.size() == input.flat_input_length;
  const bool feature_dim_valid = input.feature_dim > 0;
  last_inference_success_ = false;
  last_inference_status_ = shape_valid && flat_len_valid && data_size_valid && feature_dim_valid
                               ? "placeholder_zero_output"
                               : "placeholder_input_shape_invalid";
  (void)model_path_;
  return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

std::string PoseCompensator::OnnxPlaceholderBackend::name() const
{
  return "onnx_placeholder";
}

std::string PoseCompensator::OnnxPlaceholderBackend::statusMessage() const
{
  return model_path_.empty() ? "placeholder_zero_output_empty_model_path" : "placeholder_zero_output";
}

bool PoseCompensator::OnnxPlaceholderBackend::lastInferenceSuccess() const
{
  return last_inference_success_;
}

std::string PoseCompensator::OnnxPlaceholderBackend::lastInferenceStatus() const
{
  return last_inference_status_;
}

PoseCompensator::OnnxInferenceBackend::OnnxInferenceBackend(
    std::string model_path, std::string input_name, std::string output_name, bool use_cpu_inference)
    : impl_(std::make_unique<Impl>()),
      model_path_(std::move(model_path)),
      input_name_(std::move(input_name)),
      output_name_(std::move(output_name)),
      use_cpu_inference_(use_cpu_inference)
{
#ifdef HAVE_ONNXRUNTIME
  if (model_path_.empty())
  {
    status_message_ = "model_path_empty";
    last_error_message_ = "model_path is empty";
    return;
  }
  if (!std::filesystem::exists(model_path_))
  {
    status_message_ = "model_file_not_found";
    last_error_message_ = "model file not found: " + model_path_;
    return;
  }

  auto fail_with_message = [this](const std::string &status, const std::string &message) {
    model_loaded_ = false;
    status_message_ = status;
    last_error_message_ = message.empty() ? "none" : message;
  };

  auto fail_with_ort_exception = [this, &fail_with_message](const std::string &status, const Ort::Exception &e) {
    fail_with_message(status, e.what());
  };

  auto fail_with_std_exception = [this, &fail_with_message](const std::string &status, const std::exception &e) {
    fail_with_message(status, e.what());
  };

  auto fail_with_unknown_exception = [this, &fail_with_message](const std::string &status) {
    fail_with_message(status, "unknown_exception");
  };

  try
  {
    impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "fast_livo_pose_comp");
  }
  catch (const Ort::Exception &e)
  {
    fail_with_ort_exception("env_create_failed", e);
    return;
  }
  catch (const std::exception &e)
  {
    fail_with_std_exception("env_create_failed", e);
    return;
  }
  catch (...)
  {
    fail_with_unknown_exception("env_create_failed");
    return;
  }

  try
  {
    impl_->session_options = std::make_unique<Ort::SessionOptions>();
    impl_->session_options->SetIntraOpNumThreads(1);
    impl_->session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  }
  catch (const Ort::Exception &e)
  {
    fail_with_ort_exception("session_options_create_failed", e);
    return;
  }
  catch (const std::exception &e)
  {
    fail_with_std_exception("session_options_create_failed", e);
    return;
  }
  catch (...)
  {
    fail_with_unknown_exception("session_options_create_failed");
    return;
  }

  try
  {
    impl_->session = std::make_unique<Ort::Session>(*impl_->env, model_path_.c_str(), *impl_->session_options);
    session_ready_ = impl_->session != nullptr;
    if (!session_ready_)
    {
      fail_with_message("session_create_failed", "Ort::Session returned nullptr");
      return;
    }
    status_message_ = "session_ready";
    last_error_message_ = "none";
  }
  catch (const Ort::Exception &e)
  {
    fail_with_ort_exception("session_create_failed", e);
    return;
  }
  catch (const std::exception &e)
  {
    fail_with_std_exception("session_create_failed", e);
    return;
  }
  catch (...)
  {
    fail_with_unknown_exception("session_create_failed");
    return;
  }

  const size_t input_count = impl_->session->GetInputCount();
  const size_t output_count = impl_->session->GetOutputCount();
  if (input_count == 0)
  {
    fail_with_message("input_count_invalid", "model exposes zero inputs");
    return;
  }
  if (output_count == 0)
  {
    fail_with_message("output_count_invalid", "model exposes zero outputs");
    return;
  }

  Ort::AllocatorWithDefaultOptions allocator;
  if (input_name_.empty())
  {
    if (input_count != 1)
    {
      std::ostringstream oss;
      oss << "input_name is empty and model exposes " << input_count << " inputs";
      fail_with_message("input_name_resolve_failed", oss.str());
      return;
    }
    try
    {
      auto input_name = impl_->session->GetInputNameAllocated(0, allocator);
      input_name_ = input_name ? std::string(input_name.get()) : std::string();
    }
    catch (const Ort::Exception &e)
    {
      fail_with_ort_exception("input_name_resolve_failed", e);
      return;
    }
    catch (const std::exception &e)
    {
      fail_with_std_exception("input_name_resolve_failed", e);
      return;
    }
    catch (...)
    {
      fail_with_unknown_exception("input_name_resolve_failed");
      return;
    }
  }
  else if (findInputIndexByName(*impl_->session, input_name_) < 0)
  {
    fail_with_message("input_name_resolve_failed", "configured input_name not found: " + input_name_);
    return;
  }

  if (input_name_.empty())
  {
    fail_with_message("input_name_resolve_failed", "resolved input_name is empty");
    return;
  }

  if (output_name_.empty())
  {
    if (output_count != 1)
    {
      std::ostringstream oss;
      oss << "output_name is empty and model exposes " << output_count << " outputs";
      fail_with_message("output_name_resolve_failed", oss.str());
      return;
    }
    try
    {
      auto output_name = impl_->session->GetOutputNameAllocated(0, allocator);
      output_name_ = output_name ? std::string(output_name.get()) : std::string();
    }
    catch (const Ort::Exception &e)
    {
      fail_with_ort_exception("output_name_resolve_failed", e);
      return;
    }
    catch (const std::exception &e)
    {
      fail_with_std_exception("output_name_resolve_failed", e);
      return;
    }
    catch (...)
    {
      fail_with_unknown_exception("output_name_resolve_failed");
      return;
    }
  }
  else if (findOutputIndexByName(*impl_->session, output_name_) < 0)
  {
    fail_with_message("output_name_resolve_failed", "configured output_name not found: " + output_name_);
    return;
  }

  if (output_name_.empty())
  {
    fail_with_message("output_name_resolve_failed", "resolved output_name is empty");
    return;
  }

  const int input_index = findInputIndexByName(*impl_->session, input_name_);
  if (input_index < 0)
  {
    fail_with_message("input_name_resolve_failed", "resolved input_name not found in session: " + input_name_);
    return;
  }

  const int output_index = findOutputIndexByName(*impl_->session, output_name_);
  if (output_index < 0)
  {
    fail_with_message("output_name_resolve_failed", "resolved output_name not found in session: " + output_name_);
    return;
  }

  io_name_ready_ = true;
  status_message_ = "io_name_ready";
  last_error_message_ = "none";

  try
  {
    Ort::TypeInfo input_type_info = impl_->session->GetInputTypeInfo(static_cast<size_t>(input_index));
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    impl_->input_element_type = input_tensor_info.GetElementType();

    const size_t input_rank = input_tensor_info.GetDimensionsCount();
    if (input_rank != kExpectedInputRank)
    {
      fail_with_message(
          "input_rank_invalid",
          "expected input rank " + std::to_string(kExpectedInputRank) + ", got " + std::to_string(input_rank));
      return;
    }

    impl_->input_shape = input_tensor_info.GetShape();
  }
  catch (const Ort::Exception &e)
  {
    fail_with_ort_exception("input_type_shape_read_failed", e);
    return;
  }
  catch (const std::exception &e)
  {
    fail_with_std_exception("input_type_shape_read_failed", e);
    return;
  }
  catch (...)
  {
    fail_with_unknown_exception("input_type_shape_read_failed");
    return;
  }

  if (!isSupportedElementType(impl_->input_element_type))
  {
    fail_with_message(
        "input_type_shape_read_failed",
        "unsupported input element type: " + tensorElementTypeToString(impl_->input_element_type));
    return;
  }

  const int64_t model_seq_dim = impl_->input_shape[0];
  const int64_t model_feature_dim = impl_->input_shape[1];
  (void)model_seq_dim;
  if (model_feature_dim > 0 && model_feature_dim != kExpectedFeatureDim)
  {
    fail_with_message(
        "input_feature_dim_mismatch",
        "model feature_dim=" + std::to_string(model_feature_dim) +
            ", expected=" + std::to_string(kExpectedFeatureDim));
    return;
  }

  try
  {
    Ort::TypeInfo output_type_info = impl_->session->GetOutputTypeInfo(static_cast<size_t>(output_index));
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    impl_->output_element_type = output_tensor_info.GetElementType();

    const size_t output_rank = output_tensor_info.GetDimensionsCount();
    if (output_rank != kExpectedOutputRank)
    {
      fail_with_message(
          "output_rank_invalid",
          "expected output rank " + std::to_string(kExpectedOutputRank) + ", got " + std::to_string(output_rank));
      return;
    }

    impl_->output_shape = output_tensor_info.GetShape();
  }
  catch (const Ort::Exception &e)
  {
    fail_with_ort_exception("output_type_shape_read_failed", e);
    return;
  }
  catch (const std::exception &e)
  {
    fail_with_std_exception("output_type_shape_read_failed", e);
    return;
  }
  catch (...)
  {
    fail_with_unknown_exception("output_type_shape_read_failed");
    return;
  }

  if (!isSupportedElementType(impl_->output_element_type))
  {
    fail_with_message(
        "output_type_shape_read_failed",
        "unsupported output element type: " + tensorElementTypeToString(impl_->output_element_type));
    return;
  }

  if (impl_->output_shape[0] > 0 && impl_->output_shape[0] != kExpectedOutputDim)
  {
    fail_with_message(
        "output_dim_mismatch",
        "model output_dim=" + std::to_string(impl_->output_shape[0]) +
            ", expected=" + std::to_string(kExpectedOutputDim));
    return;
  }

  model_loaded_ = true;
  last_error_message_ = "none";
#else
  status_message_ = "onnxruntime_not_compiled";
  last_error_message_ = "onnxruntime_not_compiled";
  (void)model_path_;
  (void)input_name_;
  (void)output_name_;
  (void)use_cpu_inference_;
#endif
}

PoseCompensator::OnnxInferenceBackend::~OnnxInferenceBackend() = default;

PoseCompensator::PoseCorrection PoseCompensator::OnnxInferenceBackend::infer(const FlatFeatureInput &input) const
{
  last_inference_success_ = false;
  last_inference_status_ = "not_run";
#ifdef HAVE_ONNXRUNTIME
  if (!model_loaded_)
  {
    last_inference_status_ = "model_not_loaded";
    return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }
  if (!session_ready_ || !impl_ || !impl_->session)
  {
    last_inference_status_ = "session_not_ready";
    return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }
  if (!io_name_ready_)
  {
    last_inference_status_ = "io_name_not_ready";
    return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }

  last_error_message_ = "none";

  if (input.sequence_length == 0 || input.feature_dim == 0 || input.data.empty())
  {
    last_inference_status_ = "input_shape_mismatch";
    last_error_message_ = "sequence_length, feature_dim, and input data must all be non-zero";
    return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }

  const size_t expected_flat_size = input.sequence_length * input.feature_dim;
  if (input.flat_input_length != expected_flat_size)
  {
    last_inference_status_ = "input_flat_size_mismatch";
    last_error_message_ = "flat_input_length=" + std::to_string(input.flat_input_length) +
                          ", expected=" + std::to_string(expected_flat_size);
    return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }
  if (input.data.size() != expected_flat_size)
  {
    last_inference_status_ = "input_flat_size_mismatch";
    last_error_message_ = "input_flat.size()=" + std::to_string(input.data.size()) +
                          ", expected=" + std::to_string(expected_flat_size);
    return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }
  if (!validateModelInputShape(
          impl_->input_shape, static_cast<int64_t>(input.sequence_length), static_cast<int64_t>(input.feature_dim)))
  {
    last_inference_status_ = "input_shape_mismatch";
    last_error_message_ = inputShapeMismatchMessage(
        impl_->input_shape, static_cast<int64_t>(input.sequence_length), static_cast<int64_t>(input.feature_dim));
    return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }

  try
  {
    // Minimal first-debug assumption for the real ONNX path:
    // 1. the model has one logical sequence input with shape [T, F]
    // 2. F stays aligned with the current handcrafted feature layout
    // 3. the model outputs exactly 6 values ordered as
    //    [d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz]
    const std::array<int64_t, 2> runtime_input_shape = {
        static_cast<int64_t>(input.sequence_length),
        static_cast<int64_t>(input.feature_dim)};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    if (impl_->input_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
    {
      std::vector<double> input_tensor_values(input.data.begin(), input.data.end());
      if (input_tensor_values.size() != expected_flat_size)
      {
        last_inference_status_ = "input_flat_size_mismatch";
        last_error_message_ = "input_flat.size()=" + std::to_string(input_tensor_values.size()) +
                              ", expected=" + std::to_string(expected_flat_size);
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }
      Ort::Value input_tensor = Ort::Value::CreateTensor<double>(
          memory_info, input_tensor_values.data(), input_tensor_values.size(), runtime_input_shape.data(),
          runtime_input_shape.size());
      const char *input_names[] = {input_name_.c_str()};
      const char *output_names[] = {output_name_.c_str()};
      auto output_tensors = impl_->session->Run(
          Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
      if (output_tensors.size() != 1 || !output_tensors.front().IsTensor())
      {
        last_inference_status_ = "output_shape_mismatch";
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }

      auto output_info = output_tensors.front().GetTensorTypeAndShapeInfo();
      std::vector<int64_t> runtime_output_shape;
      std::string output_shape_error;
      if (!readExpectedRankShape(output_info, kExpectedOutputRank, "output", runtime_output_shape, output_shape_error))
      {
        last_inference_status_ = "output_shape_mismatch";
        last_error_message_ = output_shape_error;
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }
      if (runtime_output_shape[0] > 0 && runtime_output_shape[0] != kExpectedOutputDim)
      {
        last_inference_status_ = "output_shape_mismatch";
        last_error_message_ = "model output_dim=" + std::to_string(runtime_output_shape[0]) +
                              ", expected=" + std::to_string(kExpectedOutputDim);
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }
      const size_t output_count = output_info.GetElementCount();
      if (output_count != static_cast<size_t>(kExpectedOutputDim))
      {
        last_inference_status_ = "output_shape_mismatch";
        last_error_message_ = "output element_count=" + std::to_string(output_count) +
                              ", expected=" + std::to_string(kExpectedOutputDim);
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }

      const auto output_element_type = output_info.GetElementType();
      if (output_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
          output_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
      {
        last_inference_status_ = "output_type_mismatch";
        last_error_message_ = "unsupported output element type: " + tensorElementTypeToString(output_element_type);
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }

      PoseCorrection correction{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      if (output_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
      {
        const double *output_data = output_tensors.front().GetTensorData<double>();
        for (size_t i = 0; i < correction.size(); ++i)
        {
          correction[i] = output_data[i];
        }
      }
      else
      {
        const float *output_data = output_tensors.front().GetTensorData<float>();
        for (size_t i = 0; i < correction.size(); ++i)
        {
          correction[i] = static_cast<double>(output_data[i]);
        }
      }
      last_inference_success_ = true;
      last_inference_status_ = "inference_success";
      return correction;
    }
    if (impl_->input_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
      std::vector<float> input_tensor_values(input.data.begin(), input.data.end());
      if (input_tensor_values.size() != expected_flat_size)
      {
        last_inference_status_ = "input_flat_size_mismatch";
        last_error_message_ = "input_flat.size()=" + std::to_string(input_tensor_values.size()) +
                              ", expected=" + std::to_string(expected_flat_size);
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, input_tensor_values.data(), input_tensor_values.size(), runtime_input_shape.data(),
          runtime_input_shape.size());
      const char *input_names[] = {input_name_.c_str()};
      const char *output_names[] = {output_name_.c_str()};
      auto output_tensors = impl_->session->Run(
          Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
      if (output_tensors.size() != 1 || !output_tensors.front().IsTensor())
      {
        last_inference_status_ = "output_shape_mismatch";
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }

      auto output_info = output_tensors.front().GetTensorTypeAndShapeInfo();
      std::vector<int64_t> runtime_output_shape;
      std::string output_shape_error;
      if (!readExpectedRankShape(output_info, kExpectedOutputRank, "output", runtime_output_shape, output_shape_error))
      {
        last_inference_status_ = "output_shape_mismatch";
        last_error_message_ = output_shape_error;
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }
      if (runtime_output_shape[0] > 0 && runtime_output_shape[0] != kExpectedOutputDim)
      {
        last_inference_status_ = "output_shape_mismatch";
        last_error_message_ = "model output_dim=" + std::to_string(runtime_output_shape[0]) +
                              ", expected=" + std::to_string(kExpectedOutputDim);
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }
      const size_t output_count = output_info.GetElementCount();
      if (output_count != static_cast<size_t>(kExpectedOutputDim))
      {
        last_inference_status_ = "output_shape_mismatch";
        last_error_message_ = "output element_count=" + std::to_string(output_count) +
                              ", expected=" + std::to_string(kExpectedOutputDim);
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }

      const auto output_element_type = output_info.GetElementType();
      if (output_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
          output_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
      {
        last_inference_status_ = "output_type_mismatch";
        last_error_message_ = "unsupported output element type: " + tensorElementTypeToString(output_element_type);
        return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }

      PoseCorrection correction{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      if (output_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
      {
        const double *output_data = output_tensors.front().GetTensorData<double>();
        for (size_t i = 0; i < correction.size(); ++i)
        {
          correction[i] = output_data[i];
        }
      }
      else
      {
        const float *output_data = output_tensors.front().GetTensorData<float>();
        for (size_t i = 0; i < correction.size(); ++i)
        {
          correction[i] = static_cast<double>(output_data[i]);
        }
      }
      last_inference_success_ = true;
      last_inference_status_ = "inference_success";
      return correction;
    }

    last_inference_status_ = "input_type_mismatch";
    last_error_message_ = "unsupported input element type: " + tensorElementTypeToString(impl_->input_element_type);
    return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }
  catch (const Ort::Exception &e)
  {
    last_inference_status_ = "inference_exception";
    last_error_message_ = e.what();
    return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }
  catch (const std::exception &e)
  {
    last_inference_status_ = "inference_exception";
    last_error_message_ = e.what();
    return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }
  catch (...)
  {
    last_inference_status_ = "inference_exception";
    last_error_message_ = "unknown_exception";
    return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }
#else
  (void)input;
  last_inference_status_ = "onnxruntime_not_compiled";
  return PoseCorrection{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
#endif
}

std::string PoseCompensator::OnnxInferenceBackend::name() const
{
  return "onnx";
}

bool PoseCompensator::OnnxInferenceBackend::isLoaded() const
{
  return model_loaded_;
}

bool PoseCompensator::OnnxInferenceBackend::isSessionReady() const
{
  return session_ready_;
}

bool PoseCompensator::OnnxInferenceBackend::isIoNameReady() const
{
  return io_name_ready_;
}

std::string PoseCompensator::OnnxInferenceBackend::statusMessage() const
{
  return status_message_;
}

std::string PoseCompensator::OnnxInferenceBackend::errorMessage() const
{
  return last_error_message_;
}

bool PoseCompensator::OnnxInferenceBackend::lastInferenceSuccess() const
{
  return last_inference_success_;
}

std::string PoseCompensator::OnnxInferenceBackend::lastInferenceStatus() const
{
  return last_inference_status_;
}

void PoseCompensator::configure(bool enabled, int history_len, int min_ready_frames, bool debug_log_en,
                                const std::string &backend_type, const std::string &model_path,
                                const std::string &onnx_input_name, const std::string &onnx_output_name, bool use_cpu_inference,
                                double max_rotation_correction_rad, double max_translation_correction_m,
                                bool reject_non_finite_output, bool reject_oversized_output)
{
  enabled_ = enabled;
  debug_log_en_ = debug_log_en;
  history_len_ = std::max(1, history_len);
  min_ready_frames_ = std::max(1, min_ready_frames);
  backend_type_ = backend_type.empty() ? "dummy" : backend_type;
  model_path_ = model_path;
  onnx_input_name_ = onnx_input_name;
  onnx_output_name_ = onnx_output_name;
  use_cpu_inference_ = use_cpu_inference;
  max_rotation_correction_rad_ = std::max(0.0, max_rotation_correction_rad);
  max_translation_correction_m_ = std::max(0.0, max_translation_correction_m);
  reject_non_finite_output_ = reject_non_finite_output;
  reject_oversized_output_ = reject_oversized_output;
  while (history_.size() > static_cast<size_t>(history_len_))
  {
    history_.pop_front();
  }
  updateBackend();
}

void PoseCompensator::pushState(double timestamp, const StatesGroup &state, int effective_feature_num, double avg_residual)
{
  if (!enabled_) return;

  StateHistory item;
  item.timestamp = timestamp;
  item.pos_end = state.pos_end;
  item.rot_end = state.rot_end;
  item.vel_end = state.vel_end;
  item.bias_g = state.bias_g;
  item.bias_a = state.bias_a;
  item.effective_feature_num = effective_feature_num;
  item.avg_residual = avg_residual;
  history_.push_back(item);

  while (history_.size() > static_cast<size_t>(history_len_))
  {
    history_.pop_front();
  }
}

bool PoseCompensator::isEnabled() const
{
  return enabled_;
}

bool PoseCompensator::isReady() const
{
  return enabled_ && history_.size() >= static_cast<size_t>(min_ready_frames_);
}

bool PoseCompensator::debugLogEnabled() const
{
  return debug_log_en_;
}

int PoseCompensator::historyLen() const
{
  return history_len_;
}

int PoseCompensator::minReadyFrames() const
{
  return min_ready_frames_;
}

size_t PoseCompensator::historySize() const
{
  return history_.size();
}

size_t PoseCompensator::featureDimension() const
{
  // Per frame: pos(3), quaternion rot xyzw(4), vel(3), gyro bias(3),
  // accel bias(3), effective feature count(1), average residual(1).
  return 18;
}

size_t PoseCompensator::correctionDimension() const
{
  // [d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz]
  return 6;
}

size_t PoseCompensator::lastSequenceLength() const
{
  return last_sequence_length_;
}

size_t PoseCompensator::lastFeatureDimension() const
{
  return last_feature_dimension_;
}

size_t PoseCompensator::lastFlatInputLength() const
{
  return last_flat_input_length_;
}

size_t PoseCompensator::lastCorrectionDimension() const
{
  return last_correction_dimension_;
}

std::string PoseCompensator::backendName() const
{
  return backend_ ? backend_->name() : "null";
}

std::string PoseCompensator::backendType() const
{
  return backend_type_;
}

std::string PoseCompensator::modelPath() const
{
  return model_path_;
}

bool PoseCompensator::backendLoaded() const
{
  return backend_loaded_;
}

bool PoseCompensator::backendSessionReady() const
{
  return backend_ ? backend_->isSessionReady() : false;
}

bool PoseCompensator::backendIoNameReady() const
{
  return backend_ ? backend_->isIoNameReady() : false;
}

bool PoseCompensator::backendFallbackActive() const
{
  return backend_fallback_active_;
}

std::string PoseCompensator::backendFallbackReason() const
{
  return backend_fallback_reason_;
}

std::string PoseCompensator::backendStatusMessage() const
{
  return backend_status_message_;
}

std::string PoseCompensator::backendErrorMessage() const
{
  return backend_ ? backend_->errorMessage() : backend_error_message_;
}

bool PoseCompensator::lastInferenceSuccess() const
{
  return backend_ ? backend_->lastInferenceSuccess() : false;
}

std::string PoseCompensator::lastInferenceStatus() const
{
  return backend_ ? backend_->lastInferenceStatus() : "not_run";
}

bool PoseCompensator::lastCorrectionClamped() const
{
  return last_correction_clamped_;
}

bool PoseCompensator::lastCorrectionRejected() const
{
  return last_correction_rejected_;
}

std::string PoseCompensator::lastRejectReason() const
{
  return last_reject_reason_;
}

PoseCompensator::PoseCorrection PoseCompensator::lastRawCorrection() const
{
  return last_raw_correction_;
}

PoseCompensator::PoseCorrection PoseCompensator::lastSafeCorrection() const
{
  return last_safe_correction_;
}

PoseCompensator::FeatureSequence PoseCompensator::buildFeatureSequence() const
{
  FeatureSequence features;
  features.reserve(history_.size());
  for (const auto &item : history_)
  {
    Eigen::Quaterniond q(item.rot_end);
    FeatureFrame frame;
    frame.reserve(featureDimension());
    frame.push_back(item.pos_end[0]);
    frame.push_back(item.pos_end[1]);
    frame.push_back(item.pos_end[2]);
    frame.push_back(q.x());
    frame.push_back(q.y());
    frame.push_back(q.z());
    frame.push_back(q.w());
    frame.push_back(item.vel_end[0]);
    frame.push_back(item.vel_end[1]);
    frame.push_back(item.vel_end[2]);
    frame.push_back(item.bias_g[0]);
    frame.push_back(item.bias_g[1]);
    frame.push_back(item.bias_g[2]);
    frame.push_back(item.bias_a[0]);
    frame.push_back(item.bias_a[1]);
    frame.push_back(item.bias_a[2]);
    frame.push_back(static_cast<double>(item.effective_feature_num));
    frame.push_back(item.avg_residual);
    features.push_back(frame);
  }
  return features;
}

PoseCompensator::FlatFeatureInput PoseCompensator::flattenFeatureSequence(const FeatureSequence &features) const
{
  FlatFeatureInput flat_input;
  flat_input.sequence_length = features.size();
  flat_input.feature_dim = features.empty() ? 0 : features.front().size();
  flat_input.flat_input_length = flat_input.sequence_length * flat_input.feature_dim;
  flat_input.data.reserve(flat_input.flat_input_length);
  for (const auto &frame : features)
  {
    flat_input.data.insert(flat_input.data.end(), frame.begin(), frame.end());
  }
  return flat_input;
}

bool PoseCompensator::isCorrectionFinite(const PoseCorrection &correction) const
{
  for (const double value : correction)
  {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

PoseCompensator::PoseCorrection PoseCompensator::clampCorrection(const PoseCorrection &correction, bool &was_clamped) const
{
  PoseCorrection safe_correction = correction;
  was_clamped = false;
  for (size_t i = 0; i < safe_correction.size(); ++i)
  {
    const double limit = i < 3 ? max_rotation_correction_rad_ : max_translation_correction_m_;
    const double clamped_value = std::clamp(safe_correction[i], -limit, limit);
    if (clamped_value != safe_correction[i])
    {
      was_clamped = true;
      safe_correction[i] = clamped_value;
    }
  }
  return safe_correction;
}

bool PoseCompensator::shouldRejectCorrection(const PoseCorrection &raw_correction, const PoseCorrection &safe_correction,
                                             bool was_clamped, std::string &reason) const
{
  if (raw_correction.size() != correctionDimension())
  {
    reason = "dimension_mismatch";
    return true;
  }
  if (!isCorrectionFinite(raw_correction))
  {
    reason = "non_finite_output";
    return reject_non_finite_output_;
  }
  if (reject_oversized_output_ && was_clamped)
  {
    reason = "oversized_output";
    return true;
  }
  (void)safe_correction;
  reason = "none";
  return false;
}

StatesGroup PoseCompensator::applyCorrection(const StatesGroup &state, const PoseCorrection &correction) const
{
  bool is_zero = true;
  for (const double value : correction)
  {
    if (std::fabs(value) > 0.0)
    {
      is_zero = false;
      break;
    }
  }
  if (is_zero) return state;

  StatesGroup corrected_state = state;
  const double d_roll = correction[0];
  const double d_pitch = correction[1];
  const double d_yaw = correction[2];
  const V3D d_t(correction[3], correction[4], correction[5]);
  const M3D d_rot =
      (Eigen::AngleAxisd(d_yaw, V3D::UnitZ()) *
       Eigen::AngleAxisd(d_pitch, V3D::UnitY()) *
       Eigen::AngleAxisd(d_roll, V3D::UnitX()))
          .toRotationMatrix();
  corrected_state.rot_end = corrected_state.rot_end * d_rot;
  corrected_state.pos_end = corrected_state.pos_end + d_t;
  return corrected_state;
}

void PoseCompensator::updateBackend()
{
  backend_fallback_active_ = false;
  backend_fallback_reason_ = "none";
  backend_loaded_ = false;
  backend_status_message_ = "ready";
  backend_error_message_ = "none";

  if (backend_type_ == "onnx")
  {
#ifdef HAVE_ONNXRUNTIME
    backend_ = std::make_unique<OnnxInferenceBackend>(model_path_, onnx_input_name_, onnx_output_name_, use_cpu_inference_);
    backend_loaded_ = backend_->isLoaded();
    backend_status_message_ = backend_->statusMessage();
    backend_error_message_ = backend_->errorMessage();
    if (!backend_loaded_)
    {
      backend_fallback_active_ = true;
      backend_fallback_reason_ = backend_status_message_;
    }
    return;
#else
    backend_ = std::make_unique<DummyInferenceBackend>();
    backend_fallback_active_ = true;
    backend_fallback_reason_ = "onnxruntime_unavailable";
    backend_status_message_ = "onnxruntime_not_compiled";
    backend_error_message_ = "onnxruntime_not_compiled";
    return;
#endif
  }
  if (backend_type_ == "onnx_placeholder")
  {
    backend_ = std::make_unique<OnnxPlaceholderBackend>(model_path_);
    backend_status_message_ = backend_->statusMessage();
    backend_error_message_ = backend_->errorMessage();
    return;
  }
  backend_type_ = "dummy";
  backend_ = std::make_unique<DummyInferenceBackend>();
  backend_status_message_ = backend_->statusMessage();
  backend_error_message_ = backend_->errorMessage();
}

StatesGroup PoseCompensator::compensate(const StatesGroup &state) const
{
  if (!isReady() || !backend_) return state;

  const FeatureSequence features = buildFeatureSequence();
  last_sequence_length_ = features.size();
  last_feature_dimension_ = features.empty() ? 0 : features.front().size();
  const FlatFeatureInput flat_input = flattenFeatureSequence(features);
  last_flat_input_length_ = flat_input.flat_input_length;

  // Minimal ONNX debug contract:
  // 1. one logical sequence input built from recent history with shape [T, F]
  // 2. one output tensor whose total element count is exactly 6
  // 3. correction layout stays [d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz]
  const PoseCorrection correction = backend_->infer(flat_input);
  last_correction_dimension_ = correction.size();
  last_raw_correction_ = correction;
  last_correction_clamped_ = false;
  last_correction_rejected_ = false;
  last_reject_reason_ = "none";

  bool is_zero = true;
  for (const double value : correction)
  {
    if (value != 0.0)
    {
      is_zero = false;
      break;
    }
  }
  if (is_zero)
  {
    last_safe_correction_ = correction;
    return state;
  }

  bool was_clamped = false;
  const PoseCorrection safe_correction = clampCorrection(correction, was_clamped);
  last_correction_clamped_ = was_clamped;
  last_safe_correction_ = safe_correction;

  std::string reject_reason;
  if (shouldRejectCorrection(correction, safe_correction, was_clamped, reject_reason))
  {
    last_correction_rejected_ = true;
    last_reject_reason_ = reject_reason;
    return state;
  }

  return applyCorrection(state, safe_correction);
}
