#ifndef COVARIANCE_UTILS_H
#define COVARIANCE_UTILS_H

#include <Eigen/Core>

namespace fast_livo
{

// FAST-LIVO2 stores the pose error as [right/body rotation, world position].
// ROS PoseWithCovariance expects [world position, fixed/world-axis rotation].
template <int StateDimension>
Eigen::Matrix<double, 6, 6> poseCovarianceInRosOrder(
  const Eigen::Matrix<double, StateDimension, StateDimension> &state_covariance,
  const Eigen::Matrix3d &world_rotation_body)
{
  static_assert(StateDimension >= 6, "FAST-LIVO2 pose covariance requires at least six state dimensions");

  Eigen::Matrix<double, 6, 6> covariance;
  covariance.block<3, 3>(0, 0) = state_covariance.template block<3, 3>(3, 3);
  covariance.block<3, 3>(0, 3) =
    state_covariance.template block<3, 3>(3, 0) * world_rotation_body.transpose();
  covariance.block<3, 3>(3, 0) =
    world_rotation_body * state_covariance.template block<3, 3>(0, 3);
  covariance.block<3, 3>(3, 3) =
    world_rotation_body * state_covariance.template block<3, 3>(0, 0) * world_rotation_body.transpose();

  // The filter uses an algebraically equivalent non-Joseph covariance update,
  // so remove only round-off asymmetry in the published copy.
  return 0.5 * (covariance + covariance.transpose());
}

}  // namespace fast_livo

#endif  // COVARIANCE_UTILS_H
