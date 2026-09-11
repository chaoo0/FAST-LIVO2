#include "utils/covariance_utils.h"

#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include <cmath>

namespace
{

using StateCovariance = Eigen::Matrix<double, 19, 19>;

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

}  // namespace
