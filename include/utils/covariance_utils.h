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

// Propagate a LiDAR-frame point covariance and the full pose-error covariance
// into the world frame.  The FAST-LIVO2 pose error is
//   R_new = R * Exp(delta_theta_body), p_new = p + delta_p_world,
// so for p_world = R * p_imu + p the pose Jacobian is
//   [-R * skew(p_imu), I].
template <int StateDimension>
Eigen::Matrix3d worldPointCovariance(
  const Eigen::Matrix3d & lidar_point_covariance,
  const Eigen::Matrix3d & world_rotation_imu,
  const Eigen::Matrix3d & imu_rotation_lidar,
  const Eigen::Matrix3d & imu_point_cross,
  const Eigen::Matrix<double, StateDimension, StateDimension> & state_covariance)
{
  static_assert(StateDimension >= 6, "FAST-LIVO2 point covariance requires pose state blocks");

  const Eigen::Matrix3d world_rotation_lidar = world_rotation_imu * imu_rotation_lidar;
  Eigen::Matrix<double, 3, 6> pose_jacobian;
  pose_jacobian.template block<3, 3>(0, 0) = -world_rotation_imu * imu_point_cross;
  pose_jacobian.template block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

  const Eigen::Matrix3d covariance =
    world_rotation_lidar * lidar_point_covariance * world_rotation_lidar.transpose() +
    pose_jacobian * state_covariance.template block<6, 6>(0, 0) * pose_jacobian.transpose();
  return 0.5 * (covariance + covariance.transpose());
}

}  // namespace fast_livo

#endif  // COVARIANCE_UTILS_H
