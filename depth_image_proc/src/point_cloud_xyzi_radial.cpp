// Copyright (c) 2008, Willow Garage, Inc.
// All rights reserved.
//
// Software License Agreement (BSD License 2.0)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of the Willow Garage nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "depth_image_proc/visibility.h"

#include <image_transport/camera_common.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <depth_image_proc/point_cloud_xyzi_radial.hpp>
#include <depth_image_proc/depth_traits.hpp>
#include <depth_image_proc/conversions.hpp>

namespace depth_image_proc
{

PointCloudXyziRadialNode::PointCloudXyziRadialNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("PointCloudXyziRadialNode", options)
{
  // TransportHints does not actually declare the parameter
  this->declare_parameter<std::string>("image_transport", "raw");
  this->declare_parameter<std::string>("depth_image_transport", "raw");

  // Read parameters
  queue_size_ = this->declare_parameter<int>("queue_size", 5);

  // Synchronize inputs. Topic subscriptions happen on demand in the connection callback.
  sync_ = std::make_shared<Synchronizer>(
    SyncPolicy(queue_size_),
    sub_depth_,
    sub_intensity_,
    sub_info_);
  sync_->registerCallback(
    std::bind(
      &PointCloudXyziRadialNode::imageCb,
      this,
      std::placeholders::_1,
      std::placeholders::_2,
      std::placeholders::_3));

  // Create publisher with connect callback
  rclcpp::PublisherOptions pub_options;
  pub_options.event_callbacks.matched_callback =
    [this](rclcpp::MatchedInfo & s)
    {
      std::lock_guard<std::mutex> lock(connect_mutex_);
      if (s.current_count == 0) {
        sub_depth_.unsubscribe();
        sub_intensity_.unsubscribe();
        sub_info_.unsubscribe();
      } else if (!sub_depth_.getSubscriber()) {
        // For compressed topics to remap appropriately, we need to pass a
        // fully expanded and remapped topic name to image_transport
        auto node_base = this->get_node_base_interface();
        std::string depth_topic =
          node_base->resolve_topic_or_service_name("depth/image_raw", false);
        std::string intensity_topic =
          node_base->resolve_topic_or_service_name("intensity/image_raw", false);
        // Allow also remapping camera_info to something different than default
        std::string intensity_info_topic =
          node_base->resolve_topic_or_service_name(
          image_transport::getCameraInfoTopic(intensity_topic), false);

        // depth image can use different transport.(e.g. compressedDepth)
        image_transport::TransportHints depth_hints(this, "raw", "depth_image_transport");
        sub_depth_.subscribe(this, depth_topic, depth_hints.getTransport());

        // intensity uses normal ros transport hints.
        image_transport::TransportHints hints(this);
        sub_intensity_.subscribe(this, intensity_topic, hints.getTransport());
        sub_info_.subscribe(this, intensity_info_topic);
      }
    };
  pub_point_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "points", rclcpp::SensorDataQoS(), pub_options);
}

void PointCloudXyziRadialNode::imageCb(
  const Image::ConstSharedPtr & depth_msg,
  const Image::ConstSharedPtr & intensity_msg,
  const CameraInfo::ConstSharedPtr & info_msg)
{
  auto cloud_msg = std::make_shared<PointCloud>();
  cloud_msg->header = depth_msg->header;
  cloud_msg->height = depth_msg->height;
  cloud_msg->width = depth_msg->width;
  cloud_msg->is_dense = false;
  cloud_msg->is_bigendian = false;

  sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
  pcd_modifier.setPointCloud2Fields(
    4,
    "x", 1, sensor_msgs::msg::PointField::FLOAT32,
    "y", 1, sensor_msgs::msg::PointField::FLOAT32,
    "z", 1, sensor_msgs::msg::PointField::FLOAT32,
    "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);


  if (info_msg->d != D_ || info_msg->k != K_ || width_ != info_msg->width ||
    height_ != info_msg->height)
  {
    D_ = info_msg->d;
    K_ = info_msg->k;
    width_ = info_msg->width;
    height_ = info_msg->height;
    transform_ = initMatrix(cv::Mat_<double>(3, 3, &K_[0]), cv::Mat(D_), width_, height_, true);
  }

  // Convert Depth Image to Pointcloud
  if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    convertDepthRadial<uint16_t>(depth_msg, cloud_msg, transform_);
  } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    convertDepthRadial<float>(depth_msg, cloud_msg, transform_);
  } else {
    RCLCPP_ERROR(
      get_logger(), "Depth image has unsupported encoding [%s]", depth_msg->encoding.c_str());
    return;
  }

  if (intensity_msg->encoding == sensor_msgs::image_encodings::MONO8) {
    convertIntensity<uint8_t>(intensity_msg, cloud_msg);
  } else if (intensity_msg->encoding == sensor_msgs::image_encodings::MONO16) {
    convertIntensity<uint16_t>(intensity_msg, cloud_msg);
  } else if (intensity_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    convertIntensity<uint16_t>(intensity_msg, cloud_msg);
  } else if (intensity_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    convertIntensity<float>(intensity_msg, cloud_msg);
  } else {
    RCLCPP_ERROR(
      get_logger(), "Intensity image has unsupported encoding [%s]",
      intensity_msg->encoding.c_str());
    return;
  }

  pub_point_cloud_->publish(*cloud_msg);
}

}  // namespace depth_image_proc

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
RCLCPP_COMPONENTS_REGISTER_NODE(depth_image_proc::PointCloudXyziRadialNode)
