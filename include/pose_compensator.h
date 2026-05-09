/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.
*/

#ifndef POSE_COMPENSATOR_H_
#define POSE_COMPENSATOR_H_

#include "common_lib.h"
#include <array>
#include <deque>
#include <memory>
#include <string>
#include <vector>

class PoseCompensator
{
public:
  using FeatureFrame = std::vector<double>;
  using FeatureSequence = std::vector<FeatureFrame>;
  using PoseCorrection = std::array<double, 6>;

  struct FlatFeatureInput
  {
    std::vector<double> data;
    size_t sequence_length = 0;
    size_t feature_dim = 0;
    size_t flat_input_length = 0;
  };

  struct StateHistory
  {
    double timestamp = 0.0;
    V3D pos_end = V3D::Zero();
    M3D rot_end = M3D::Identity();
    V3D vel_end = V3D::Zero();
    V3D bias_g = V3D::Zero();
    V3D bias_a = V3D::Zero();
    int effective_feature_num = 0;
    double avg_residual = -1.0;
  };

  class InferenceBackend
  {
  public:
    virtual ~InferenceBackend() = default;
    virtual PoseCorrection infer(const FlatFeatureInput &input) const = 0;
    virtual std::string name() const = 0;
    virtual bool isLoaded() const { return false; }
    virtual bool isSessionReady() const { return false; }
    virtual bool isIoNameReady() const { return false; }
    virtual std::string statusMessage() const { return "ready"; }
    virtual std::string errorMessage() const { return "none"; }
    virtual bool lastInferenceSuccess() const { return false; }
    virtual std::string lastInferenceStatus() const { return "not_run"; }
  };

  class DummyInferenceBackend final : public InferenceBackend
  {
  public:
    PoseCorrection infer(const FlatFeatureInput &input) const override;
    std::string name() const override;
    std::string statusMessage() const override;
    bool lastInferenceSuccess() const override;
    std::string lastInferenceStatus() const override;

  private:
    mutable bool last_inference_success_ = false;
    mutable std::string last_inference_status_ = "dummy_zero_output";
  };

  class OnnxPlaceholderBackend final : public InferenceBackend
  {
  public:
    explicit OnnxPlaceholderBackend(std::string model_path);
    PoseCorrection infer(const FlatFeatureInput &input) const override;
    std::string name() const override;
    std::string statusMessage() const override;
    bool lastInferenceSuccess() const override;
    std::string lastInferenceStatus() const override;

  private:
    std::string model_path_;
    mutable bool last_inference_success_ = false;
    mutable std::string last_inference_status_ = "placeholder_zero_output";
  };

  class OnnxInferenceBackend final : public InferenceBackend
  {
  public:
    OnnxInferenceBackend(std::string model_path, std::string input_name, std::string output_name, bool use_cpu_inference);
    ~OnnxInferenceBackend() override;
    PoseCorrection infer(const FlatFeatureInput &input) const override;
    std::string name() const override;
    bool isLoaded() const override;
    bool isSessionReady() const override;
    bool isIoNameReady() const override;
    std::string statusMessage() const override;
    std::string errorMessage() const override;
    bool lastInferenceSuccess() const override;
    std::string lastInferenceStatus() const override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string model_path_;
    std::string input_name_;
    std::string output_name_;
    bool use_cpu_inference_ = true;
    bool model_loaded_ = false;
    bool session_ready_ = false;
    bool io_name_ready_ = false;
    std::string status_message_ = "not_initialized";
    mutable bool last_inference_success_ = false;
    mutable std::string last_inference_status_ = "not_run";
    mutable std::string last_error_message_ = "none";
  };

  PoseCompensator();
  ~PoseCompensator() = default;

  void configure(bool enabled, int history_len, int min_ready_frames, bool debug_log_en,
                 const std::string &backend_type, const std::string &model_path,
                 const std::string &onnx_input_name, const std::string &onnx_output_name, bool use_cpu_inference,
                 double max_rotation_correction_rad, double max_translation_correction_m,
                 bool reject_non_finite_output, bool reject_oversized_output);
  void pushState(double timestamp, const StatesGroup &state, int effective_feature_num, double avg_residual);
  bool isEnabled() const;
  bool isReady() const;
  bool debugLogEnabled() const;
  int historyLen() const;
  int minReadyFrames() const;
  size_t historySize() const;
  size_t featureDimension() const;
  size_t correctionDimension() const;
  size_t lastSequenceLength() const;
  size_t lastFeatureDimension() const;
  size_t lastFlatInputLength() const;
  size_t lastCorrectionDimension() const;
  std::string backendName() const;
  std::string backendType() const;
  std::string modelPath() const;
  bool backendLoaded() const;
  bool backendSessionReady() const;
  bool backendIoNameReady() const;
  bool backendFallbackActive() const;
  std::string backendFallbackReason() const;
  std::string backendStatusMessage() const;
  std::string backendErrorMessage() const;
  bool lastInferenceSuccess() const;
  std::string lastInferenceStatus() const;
  bool lastCorrectionClamped() const;
  bool lastCorrectionRejected() const;
  std::string lastRejectReason() const;
  PoseCorrection lastRawCorrection() const;
  PoseCorrection lastSafeCorrection() const;

  // This path always routes correction through backend inference, validation,
  // and the existing safety guard before touching the state.
  StatesGroup compensate(const StatesGroup &state) const;

private:
  FeatureSequence buildFeatureSequence() const;
  FlatFeatureInput flattenFeatureSequence(const FeatureSequence &features) const;
  bool isCorrectionFinite(const PoseCorrection &correction) const;
  PoseCorrection clampCorrection(const PoseCorrection &correction, bool &was_clamped) const;
  bool shouldRejectCorrection(const PoseCorrection &raw_correction, const PoseCorrection &safe_correction,
                              bool was_clamped, std::string &reason) const;
  StatesGroup applyCorrection(const StatesGroup &state, const PoseCorrection &correction) const;
  void updateBackend();

  bool enabled_ = false;
  bool debug_log_en_ = false;
  int history_len_ = 10;
  int min_ready_frames_ = 3;
  std::string backend_type_ = "dummy";
  std::string model_path_;
  std::string onnx_input_name_;
  std::string onnx_output_name_;
  bool use_cpu_inference_ = true;
  double max_rotation_correction_rad_ = 0.10;
  double max_translation_correction_m_ = 0.20;
  bool reject_non_finite_output_ = true;
  bool reject_oversized_output_ = false;
  std::deque<StateHistory> history_;
  std::unique_ptr<InferenceBackend> backend_;
  bool backend_loaded_ = false;
  bool backend_fallback_active_ = false;
  std::string backend_fallback_reason_ = "none";
  std::string backend_status_message_ = "not_initialized";
  std::string backend_error_message_ = "none";
  mutable size_t last_sequence_length_ = 0;
  mutable size_t last_feature_dimension_ = 0;
  mutable size_t last_flat_input_length_ = 0;
  mutable size_t last_correction_dimension_ = 0;
  mutable bool last_correction_clamped_ = false;
  mutable bool last_correction_rejected_ = false;
  mutable std::string last_reject_reason_ = "none";
  mutable PoseCorrection last_raw_correction_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  mutable PoseCorrection last_safe_correction_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

#endif // POSE_COMPENSATOR_H_
