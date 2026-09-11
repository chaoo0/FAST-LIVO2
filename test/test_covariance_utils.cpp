#include "utils/covariance_utils.h"

#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include <cmath>

namespace
{

using StateCovariance = Eigen::Matrix<double, 19, 19>;

Eigen::Matrix3d rotationExp(const Eigen::Vector3d & rotation_vector)
{
  const double angle = rotation_vector.norm();
  if (angle == 0.0) {
    return Eigen::Matrix3d::Identity();
  }
  return Eigen::AngleAxisd(angle, rotation_vector / angle).toRotationMatrix();
}

TEST(PoseCovarianceInRosOrder, ReordersRotationAndPositionAtIdentity)
{
  StateCovariance state_covariance = StateCovariance::Zero();
  state_covariance.diagonal().head<3>() << 1.0, 2.0, 3.0;
  state_covariance.diagonal().segment<3>(3) << 4.0, 5.0, 6.0;

  const auto covariance =
    fast_livo::poseCovarianceInRosOrder(state_covariance, Eigen::Matrix3d::Identity());

  const Eigen::Matrix<double, 6, 1> expected =
    (Eigen::Matrix<double, 6, 1>() << 4.0, 5.0, 6.0, 1.0, 2.0, 3.0).finished();
  EXPECT_TRUE(covariance.diagonal().isApprox(expected));
}

TEST(PoseCovarianceInRosOrder, RotatesBodyTangentIntoFixedWorldAxes)
{
  StateCovariance state_covariance = StateCovariance::Zero();
  state_covariance.diagonal().head<3>() << 1.0, 2.0, 3.0;
  state_covariance.diagonal().segment<3>(3) << 4.0, 5.0, 6.0;

  constexpr double half_pi = 1.57079632679489661923;
  Eigen::Matrix3d world_rotation_body;
  world_rotation_body = Eigen::AngleAxisd(half_pi, Eigen::Vector3d::UnitZ());
  const auto covariance = fast_livo::poseCovarianceInRosOrder(state_covariance, world_rotation_body);

  const Eigen::Matrix<double, 6, 1> expected =
    (Eigen::Matrix<double, 6, 1>() << 4.0, 5.0, 6.0, 2.0, 1.0, 3.0).finished();
  EXPECT_TRUE(covariance.diagonal().isApprox(expected, 1e-12));
}

TEST(PoseCovarianceInRosOrder, TransformsCrossCovarianceAndPublishesSymmetrically)
{
  StateCovariance state_covariance = StateCovariance::Identity();
  Eigen::Matrix3d position_rotation_cross;
  position_rotation_cross <<
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6,
    0.7, 0.8, 0.9;
  state_covariance.block<3, 3>(3, 0) = position_rotation_cross;
  state_covariance.block<3, 3>(0, 3) = position_rotation_cross.transpose();

  constexpr double angle = 0.37;
  const Eigen::Matrix3d world_rotation_body =
    Eigen::AngleAxisd(angle, Eigen::Vector3d(1.0, 2.0, 3.0).normalized()).toRotationMatrix();
  const auto covariance = fast_livo::poseCovarianceInRosOrder(state_covariance, world_rotation_body);
  const Eigen::Matrix3d transformed_cross = covariance.block<3, 3>(0, 3);
  const Eigen::Matrix3d expected_cross = position_rotation_cross * world_rotation_body.transpose();

  EXPECT_TRUE(transformed_cross.isApprox(expected_cross, 1e-12));
  EXPECT_TRUE(covariance.isApprox(covariance.transpose(), 1e-12));
}

TEST(WorldPointCovariance, MatchesTheRightPerturbationJacobian)
{
  StateCovariance state_covariance = StateCovariance::Zero();
  state_covariance.diagonal().head<6>() << 0.01, 0.02, 0.03, 0.04, 0.05, 0.06;
  Eigen::Matrix3d rotation_position_cross;
  rotation_position_cross <<
    0.001, -0.002, 0.003,
    0.002, 0.001, -0.001,
    -0.003, 0.002, 0.001;
  state_covariance.block<3, 3>(0, 3) = rotation_position_cross;
  state_covariance.block<3, 3>(3, 0) = rotation_position_cross.transpose();

  const Eigen::Matrix3d world_rotation_imu =
    Eigen::AngleAxisd(0.61, Eigen::Vector3d(1.0, -2.0, 0.5).normalized()).toRotationMatrix();
  const Eigen::Matrix3d imu_rotation_lidar =
    Eigen::AngleAxisd(-0.24, Eigen::Vector3d(0.2, 0.7, 1.0).normalized()).toRotationMatrix();
  const Eigen::Vector3d point_imu(2.0, -0.5, 1.2);
  Eigen::Matrix3d point_cross;
  point_cross <<
    0.0, -point_imu.z(), point_imu.y(),
    point_imu.z(), 0.0, -point_imu.x(),
    -point_imu.y(), point_imu.x(), 0.0;
  const Eigen::Matrix3d lidar_covariance =
    (Eigen::Vector3d(0.004, 0.006, 0.009)).asDiagonal();

  Eigen::Matrix<double, 3, 6> pose_jacobian;
  pose_jacobian.block<3, 3>(0, 0) = -world_rotation_imu * point_cross;
  pose_jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d world_rotation_lidar = world_rotation_imu * imu_rotation_lidar;
  const Eigen::Matrix3d expected =
    world_rotation_lidar * lidar_covariance * world_rotation_lidar.transpose() +
    pose_jacobian * state_covariance.block<6, 6>(0, 0) * pose_jacobian.transpose();

  const auto covariance = fast_livo::worldPointCovariance(
    lidar_covariance, world_rotation_imu, imu_rotation_lidar, point_cross, state_covariance);
  EXPECT_TRUE(covariance.isApprox(expected, 1e-12));
  EXPECT_TRUE(covariance.isApprox(covariance.transpose(), 1e-12));
}

TEST(WorldPointCovariance, UsesTheNumericalRightPerturbationJacobian)
{
  const Eigen::Matrix3d world_rotation_imu =
    Eigen::AngleAxisd(0.61, Eigen::Vector3d(1.0, -2.0, 0.5).normalized()).toRotationMatrix();
  const Eigen::Vector3d point_imu(2.0, -0.5, 1.2);
  Eigen::Matrix3d point_cross;
  point_cross <<
    0.0, -point_imu.z(), point_imu.y(),
    point_imu.z(), 0.0, -point_imu.x(),
    -point_imu.y(), point_imu.x(), 0.0;

  Eigen::Matrix<double, 3, 6> numerical_jacobian;
  constexpr double epsilon = 1e-7;
  for (int column = 0; column < 6; ++column) {
    Eigen::Matrix<double, 6, 1> positive = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> negative = Eigen::Matrix<double, 6, 1>::Zero();
    positive(column) = epsilon;
    negative(column) = -epsilon;
    const Eigen::Vector3d point_positive =
      world_rotation_imu * rotationExp(positive.head<3>()) * point_imu + positive.tail<3>();
    const Eigen::Vector3d point_negative =
      world_rotation_imu * rotationExp(negative.head<3>()) * point_imu + negative.tail<3>();
    numerical_jacobian.col(column) = (point_positive - point_negative) / (2.0 * epsilon);
  }

  Eigen::Matrix<double, 3, 6> expected_jacobian;
  expected_jacobian.block<3, 3>(0, 0) = -world_rotation_imu * point_cross;
  expected_jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  EXPECT_TRUE(numerical_jacobian.isApprox(expected_jacobian, 1e-8));
}

TEST(WorldPointCovariance, IsEquivariantToAWorldFrameRotation)
{
  StateCovariance state_covariance = StateCovariance::Zero();
  state_covariance.diagonal().head<6>() << 0.02, 0.01, 0.03, 0.05, 0.04, 0.06;
  state_covariance.block<3, 3>(0, 3) <<
    0.001, 0.002, -0.001,
    -0.002, 0.001, 0.003,
    0.001, -0.003, 0.002;
  state_covariance.block<3, 3>(3, 0) = state_covariance.block<3, 3>(0, 3).transpose();

  const Eigen::Matrix3d world_rotation_imu =
    Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Matrix3d imu_rotation_lidar =
    Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitX()).toRotationMatrix();
  const Eigen::Vector3d point_imu(1.5, 0.7, -0.4);
  Eigen::Matrix3d point_cross;
  point_cross <<
    0.0, -point_imu.z(), point_imu.y(),
    point_imu.z(), 0.0, -point_imu.x(),
    -point_imu.y(), point_imu.x(), 0.0;
  const Eigen::Matrix3d lidar_covariance =
    (Eigen::Vector3d(0.003, 0.005, 0.008)).asDiagonal();
  const Eigen::Matrix3d world_basis_rotation =
    Eigen::AngleAxisd(0.73, Eigen::Vector3d(1.0, 1.0, 2.0).normalized()).toRotationMatrix();

  const Eigen::Matrix3d covariance = fast_livo::worldPointCovariance(
    lidar_covariance, world_rotation_imu, imu_rotation_lidar, point_cross, state_covariance);

  StateCovariance rotated_state_covariance = state_covariance;
  rotated_state_covariance.block<3, 3>(3, 3) =
    world_basis_rotation * state_covariance.block<3, 3>(3, 3) * world_basis_rotation.transpose();
  rotated_state_covariance.block<3, 3>(0, 3) =
    state_covariance.block<3, 3>(0, 3) * world_basis_rotation.transpose();
  rotated_state_covariance.block<3, 3>(3, 0) =
    world_basis_rotation * state_covariance.block<3, 3>(3, 0);
  const Eigen::Matrix3d rotated_covariance = fast_livo::worldPointCovariance(
    lidar_covariance, world_basis_rotation * world_rotation_imu, imu_rotation_lidar,
    point_cross, rotated_state_covariance);

  EXPECT_TRUE(rotated_covariance.isApprox(
    world_basis_rotation * covariance * world_basis_rotation.transpose(), 1e-12));
}

}  // namespace
