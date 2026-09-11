#ifdef PRE_ROS_IRON
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif

#include <gtest/gtest.h>
#include <sensor_msgs/image_encodings.hpp>

#include <cstdint>
#include <memory>

namespace
{

sensor_msgs::msg::Image::SharedPtr makeBgrImage()
{
  auto message = std::make_shared<sensor_msgs::msg::Image>();
  message->height = 1;
  message->width = 2;
  message->encoding = sensor_msgs::image_encodings::BGR8;
  message->is_bigendian = false;
  message->step = 6;
  message->data = {1, 2, 3, 4, 5, 6};
  return message;
}

TEST(CvBridgeLifetime, SharedMatDoesNotRetainTheRosMessageOwner)
{
  auto message = makeBgrImage();
  const std::weak_ptr<sensor_msgs::msg::Image> weak_message = message;
  const std::uint8_t *const source_data = message->data.data();

  const cv::Mat shared_image = cv_bridge::toCvShare(message, "bgr8")->image;

  EXPECT_EQ(shared_image.data, source_data);
  message.reset();
  EXPECT_TRUE(weak_message.expired());
  // Do not dereference shared_image here: its external storage has expired.
}

TEST(CvBridgeLifetime, CopiedMatOwnsPixelsAfterTheRosMessageExpires)
{
  auto message = makeBgrImage();
  const std::weak_ptr<sensor_msgs::msg::Image> weak_message = message;
  const std::uint8_t *const source_data = message->data.data();

  const cv::Mat copied_image = cv_bridge::toCvCopy(message, "bgr8")->image;

  EXPECT_NE(copied_image.data, source_data);
  message.reset();
  EXPECT_TRUE(weak_message.expired());
  ASSERT_EQ(copied_image.rows, 1);
  ASSERT_EQ(copied_image.cols, 2);
  EXPECT_EQ(copied_image.at<cv::Vec3b>(0, 0), cv::Vec3b(1, 2, 3));
  EXPECT_EQ(copied_image.at<cv::Vec3b>(0, 1), cv::Vec3b(4, 5, 6));
}

}  // namespace
