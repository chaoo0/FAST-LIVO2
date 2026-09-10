#include "LIVMapper.h"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);

  auto nh = std::make_shared<rclcpp::Node>("laserMapping", options);
  image_transport::ImageTransport it_(nh);
  LIVMapper mapper(nh);
  mapper.initializeSubscribersAndPublishers(it_);
  mapper.run();
  rclcpp::shutdown();
  return 0;
}
