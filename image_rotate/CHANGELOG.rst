^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package image_rotate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

5.0.4 (2024-08-20)
------------------
* Finish QoS updates (backport `#1019 <https://github.com/ros-perception/image_pipeline/issues/1019>`_) (`#1024 <https://github.com/ros-perception/image_pipeline/issues/1024>`_)
  This implements the remainder of `#847 <https://github.com/ros-perception/image_pipeline/issues/847>`_:
  - Make sure publishers default to system defaults (reliable)
  - Add QoS overriding where possible (some of the image_transport /
  message_filters stuff doesn't really support that)
  - Use the matching heuristic for subscribers consistently
* Contributors: mergify[bot]

5.0.3 (2024-07-16)
------------------

5.0.2 (2024-05-27)
------------------

5.0.1 (2024-03-26)
------------------
* fix image publisher remapping (`#941 <https://github.com/ros-perception/image_pipeline/issues/941>`_)
  Addresses `#940 <https://github.com/ros-perception/image_pipeline/issues/940>`_ - fixes the compressed/etc topic remapping for publishers
* unified changelog, add missing image, deduplicate tutorials (`#938 <https://github.com/ros-perception/image_pipeline/issues/938>`_)
  Last bit of documentation updates - putting together a single changelog
  summary for the whole release (rather than scattering among packages).
  Unified the camera_info tutorial so it isn't duplicated. Added a missing
  image from image_rotate (was on local disk, but hadn't committed it)
* add docs for image_rotate/publisher (`#936 <https://github.com/ros-perception/image_pipeline/issues/936>`_)
* Contributors: Michael Ferguson

5.0.0 (2024-01-24)
------------------
* Removed cfg files related with ROS 1 parameters (`#911 <https://github.com/ros-perception/image_pipeline/issues/911>`_)
  Removed cfg files related with ROS 1 parameters
* image_rotate: clean up (`#862 <https://github.com/ros-perception/image_pipeline/issues/862>`_)
  This is the first component/node with a cleanup pass to be fully
  implemented:
  * Fix `#740 <https://github.com/ros-perception/image_pipeline/issues/740>`_ by initializing vectors. Do this by declaring parameters
  AFTER we define the callback
  * Implement lazy subscribers (I missed this in the earlier PRs)
  * Add image_transport parameter so we can specify that desired transport
  of our subscriptions
  * Update how we test for connectivity (and update the debug message -
  has nothing to do with whether we are remapped, it's really about
  whether we are connected)
* load image_rotate::ImageRotateNode as component (`#855 <https://github.com/ros-perception/image_pipeline/issues/855>`_)
  This is a fixed version of `#820 <https://github.com/ros-perception/image_pipeline/issues/820>`_ - targeting rolling
* add myself as a maintainer (`#846 <https://github.com/ros-perception/image_pipeline/issues/846>`_)
* Contributors: Alejandro Hernández Cordero, Michael Ferguson

3.0.1 (2022-12-04)
------------------
* Replace deprecated headers
  Fixing compiler warnings.
* Contributors: Jacob Perron

3.0.0 (2022-04-29)
------------------
* Cleanup the image_rotate package.
* Replace deprecated geometry2 headers
* revert a293252
* Replace deprecated geometry2 headers
* Add maintainer (`#667 <https://github.com/ros-perception/image_pipeline/issues/667>`_)
* Contributors: Chris Lalancette, Jacob Perron, Patrick Musau

2.2.1 (2020-08-27)
------------------
* remove email blasts from steve macenski (`#596 <https://github.com/ros-perception/image_pipeline/issues/596>`_)
* [Foxy] Use ament_auto Macros (`#573 <https://github.com/ros-perception/image_pipeline/issues/573>`_)
* Contributors: Joshua Whitley, Steve Macenski

2.2.0 (2020-07-27)
------------------
* Replacing deprecated header includes with new HPP versions. (`#566 <https://github.com/ros-perception/image_pipeline/issues/566>`_)
* Use newer 'add_on_set_parameters_callback' API (`#562 <https://github.com/ros-perception/image_pipeline/issues/562>`_)
  The old API was deprecated in Foxy and since removed in https://github.com/ros2/rclcpp/pull/1199.
* Contributors: Jacob Perron, Joshua Whitley

* Initial ROS2 commit.
* Contributors: Michael Carroll

1.12.23 (2018-05-10)
--------------------

1.12.22 (2017-12-08)
--------------------

1.12.21 (2017-11-05)
--------------------
* [image_rotate] Added TF timeout so that transforms only need to be newer than last frame. (`#293 <https://github.com/ros-perception/image_pipeline/issues/293>`_)
* Contributors: mhosmar-cpr

1.12.20 (2017-04-30)
--------------------
* Fix CMake warnings about Eigen.
* address gcc6 build error
  With gcc6, compiling fails with `stdlib.h: No such file or directory`,
  as including '-isystem /usr/include' breaks with gcc6, cf.,
  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70129.
  This commit addresses this issue for this package in the same way
  it was addressed in various other ROS packages. A list of related
  commits and pull requests is at:
  https://github.com/ros/rosdistro/issues/12783
  Signed-off-by: Lukas Bulwahn <lukas.bulwahn@oss.bmw-carit.de>
* Contributors: Lukas Bulwahn, Vincent Rabaud

1.12.19 (2016-07-24)
--------------------
* Fix frames if it is empty to rotate image
* Contributors: Kentaro Wada

1.12.18 (2016-07-12)
--------------------

1.12.17 (2016-07-11)
--------------------

1.12.16 (2016-03-19)
--------------------
* clean OpenCV dependency in package.xml
* Contributors: Vincent Rabaud

1.12.15 (2016-01-17)
--------------------

1.12.14 (2015-07-22)
--------------------

1.12.13 (2015-04-06)
--------------------

1.12.12 (2014-12-31)
--------------------

1.12.11 (2014-10-26)
--------------------

1.12.10 (2014-09-28)
--------------------

1.12.9 (2014-09-21)
-------------------

1.12.8 (2014-08-19)
-------------------

1.12.6 (2014-07-27)
-------------------

1.12.4 (2014-04-28)
-------------------

1.12.3 (2014-04-12)
-------------------

1.12.2 (2014-04-08)
-------------------
* use NODELET_** macros instead of ROS_** macros
* use getNodeHandle rather than getPrivateNodeHandle
* add executable to load image_rotate/image_rotate nodelet.
  add xml file to export nodelet definition.
  Conflicts:
  image_rotate/package.xml
* make image_rotate nodelet class
  Conflicts:
  image_rotate/CMakeLists.txt
  image_rotate/package.xml
  image_rotate/src/nodelet/image_rotate_nodelet.cpp
* move image_rotate.cpp to nodelet directory according to the directory convenstion of image_pipeline
* Contributors: Ryohei Ueda

1.12.1 (2014-04-06)
-------------------
* replace tf usage by tf2 usage
