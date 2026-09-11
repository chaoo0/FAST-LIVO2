#include "preprocess.h"

#include <gtest/gtest.h>
#include <pcl_conversions/pcl_conversions.h>

#include <memory>

namespace
{

template<typename PointT>
sensor_msgs::msg::PointCloud2::ConstSharedPtr toMessage(const pcl::PointCloud<PointT> & cloud)
{
  auto message = std::make_shared<sensor_msgs::msg::PointCloud2>();
  pcl::toROSMsg(cloud, *message);
  return message;
}

TEST(PreprocessTimestamps, SortsOusterOffsetsForEndTimeAndBackwardUndistortion)
{
  pcl::PointCloud<ouster_ros::Point> input;
  input.resize(3);
  input[0].x = 1.0F;
  input[0].t = 0U;
  input[1].x = 1.0F;
  input[1].t = 20000000U;
  input[2].x = 1.0F;
  input[2].t = 10000000U;

  Preprocess preprocess;
  preprocess.set(false, OUST64, 0.1, 1);
  PointCloudXYZI::Ptr output(new PointCloudXYZI());
  preprocess.process(toMessage(input), output);

  ASSERT_EQ(output->size(), 3U);
  EXPECT_DOUBLE_EQ(output->at(0).curvature, 0.0);
  EXPECT_DOUBLE_EQ(output->at(1).curvature, 10.0);
  EXPECT_DOUBLE_EQ(output->at(2).curvature, 20.0);
}

TEST(PreprocessTimestamps, PreservesPandarFieldsAndUsesRelativeScanTime)
{
  pcl::PointCloud<Pandar128_ros::Point> input;
  input.resize(3);
  input[0].x = 1.0F;
  input[0].intensity = 51U;
  input[0].timestamp = 100.0;
  input[0].ring = 7U;
  input[1].x = 1.0F;
  input[1].intensity = 255U;
  input[1].timestamp = 100.02;
  input[1].ring = 8U;
  input[2].x = 1.0F;
  input[2].intensity = 102U;
  input[2].timestamp = 100.01;
  input[2].ring = 9U;

  Preprocess preprocess;
  preprocess.set(false, PANDAR128, 0.1, 1);
  PointCloudXYZI::Ptr output(new PointCloudXYZI());
  preprocess.process(toMessage(input), output);

  ASSERT_EQ(output->size(), 3U);
  EXPECT_NEAR(output->at(0).curvature, 0.0, 1.0e-9);
  EXPECT_NEAR(output->at(1).curvature, 10.0, 1.0e-9);
  EXPECT_NEAR(output->at(2).curvature, 20.0, 1.0e-9);
  EXPECT_NEAR(output->at(0).intensity, 0.2, 1.0e-6);
  EXPECT_NEAR(output->at(1).intensity, 0.4, 1.0e-6);
  EXPECT_NEAR(output->at(2).intensity, 1.0, 1.0e-6);
}

TEST(PreprocessTimestamps, AcceptsEmptyPandarCloud)
{
  pcl::PointCloud<Pandar128_ros::Point> input;
  Preprocess preprocess;
  preprocess.set(false, PANDAR128, 0.1, 1);
  PointCloudXYZI::Ptr output(new PointCloudXYZI());

  EXPECT_NO_THROW(preprocess.process(toMessage(input), output));
  EXPECT_TRUE(output->empty());
}

}  // namespace
