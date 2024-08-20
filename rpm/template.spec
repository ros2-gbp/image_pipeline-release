%bcond_without tests
%bcond_without weak_deps

%global __os_install_post %(echo '%{__os_install_post}' | sed -e 's!/usr/lib[^[:space:]]*/brp-python-bytecompile[[:space:]].*$!!g')
%global __provides_exclude_from ^/opt/ros/jazzy/.*$
%global __requires_exclude_from ^/opt/ros/jazzy/.*$

Name:           ros-jazzy-stereo-image-proc
Version:        5.0.4
Release:        1%{?dist}%{?release_suffix}
Summary:        ROS stereo_image_proc package

License:        BSD
URL:            https://index.ros.org/p/stereo_image_proc/github-ros-perception-image_pipeline/
Source0:        %{name}-%{version}.tar.gz

Requires:       ros-jazzy-cv-bridge
Requires:       ros-jazzy-image-geometry
Requires:       ros-jazzy-image-proc
Requires:       ros-jazzy-image-transport
Requires:       ros-jazzy-message-filters
Requires:       ros-jazzy-rclcpp
Requires:       ros-jazzy-rclcpp-components
Requires:       ros-jazzy-sensor-msgs
Requires:       ros-jazzy-stereo-msgs
Requires:       ros-jazzy-ros-workspace
BuildRequires:  ros-jazzy-ament-cmake-auto
BuildRequires:  ros-jazzy-cv-bridge
BuildRequires:  ros-jazzy-image-geometry
BuildRequires:  ros-jazzy-image-proc
BuildRequires:  ros-jazzy-image-transport
BuildRequires:  ros-jazzy-message-filters
BuildRequires:  ros-jazzy-rclcpp
BuildRequires:  ros-jazzy-rclcpp-components
BuildRequires:  ros-jazzy-sensor-msgs
BuildRequires:  ros-jazzy-stereo-msgs
BuildRequires:  ros-jazzy-ros-workspace
Provides:       %{name}-devel = %{version}-%{release}
Provides:       %{name}-doc = %{version}-%{release}
Provides:       %{name}-runtime = %{version}-%{release}

%if 0%{?with_tests}
BuildRequires:  python3-opencv
BuildRequires:  ros-jazzy-ament-cmake-pytest
BuildRequires:  ros-jazzy-ament-lint-auto
BuildRequires:  ros-jazzy-ament-lint-common
BuildRequires:  ros-jazzy-launch
BuildRequires:  ros-jazzy-launch-ros
BuildRequires:  ros-jazzy-launch-testing
BuildRequires:  ros-jazzy-launch-testing-ament-cmake
BuildRequires:  ros-jazzy-python-cmake-module
BuildRequires:  ros-jazzy-rclpy
BuildRequires:  ros-jazzy-ros-testing
%endif

%description
Stereo and single image rectification and disparity processing.

%prep
%autosetup -p1

%build
# In case we're installing to a non-standard location, look for a setup.sh
# in the install tree and source it.  It will set things like
# CMAKE_PREFIX_PATH, PKG_CONFIG_PATH, and PYTHONPATH.
if [ -f "/opt/ros/jazzy/setup.sh" ]; then . "/opt/ros/jazzy/setup.sh"; fi
mkdir -p .obj-%{_target_platform} && cd .obj-%{_target_platform}
%cmake3 \
    -UINCLUDE_INSTALL_DIR \
    -ULIB_INSTALL_DIR \
    -USYSCONF_INSTALL_DIR \
    -USHARE_INSTALL_PREFIX \
    -ULIB_SUFFIX \
    -DCMAKE_INSTALL_PREFIX="/opt/ros/jazzy" \
    -DAMENT_PREFIX_PATH="/opt/ros/jazzy" \
    -DCMAKE_PREFIX_PATH="/opt/ros/jazzy" \
    -DSETUPTOOLS_DEB_LAYOUT=OFF \
%if !0%{?with_tests}
    -DBUILD_TESTING=OFF \
%endif
    ..

%make_build

%install
# In case we're installing to a non-standard location, look for a setup.sh
# in the install tree and source it.  It will set things like
# CMAKE_PREFIX_PATH, PKG_CONFIG_PATH, and PYTHONPATH.
if [ -f "/opt/ros/jazzy/setup.sh" ]; then . "/opt/ros/jazzy/setup.sh"; fi
%make_install -C .obj-%{_target_platform}

%if 0%{?with_tests}
%check
# Look for a Makefile target with a name indicating that it runs tests
TEST_TARGET=$(%__make -qp -C .obj-%{_target_platform} | sed "s/^\(test\|check\):.*/\\1/;t f;d;:f;q0")
if [ -n "$TEST_TARGET" ]; then
# In case we're installing to a non-standard location, look for a setup.sh
# in the install tree and source it.  It will set things like
# CMAKE_PREFIX_PATH, PKG_CONFIG_PATH, and PYTHONPATH.
if [ -f "/opt/ros/jazzy/setup.sh" ]; then . "/opt/ros/jazzy/setup.sh"; fi
CTEST_OUTPUT_ON_FAILURE=1 \
    %make_build -C .obj-%{_target_platform} $TEST_TARGET || echo "RPM TESTS FAILED"
else echo "RPM TESTS SKIPPED"; fi
%endif

%files
/opt/ros/jazzy

%changelog
* Tue Aug 20 2024 Vincent Rabaud <vincent.rabaud@gmail.com> - 5.0.4-1
- Autogenerated by Bloom

* Tue Jul 16 2024 Vincent Rabaud <vincent.rabaud@gmail.com> - 6.0.0-1
- Autogenerated by Bloom

* Tue Jul 16 2024 Vincent Rabaud <vincent.rabaud@gmail.com> - 5.0.3-1
- Autogenerated by Bloom

* Mon May 27 2024 Vincent Rabaud <vincent.rabaud@gmail.com> - 5.0.1-3
- Autogenerated by Bloom

* Thu Apr 18 2024 Vincent Rabaud <vincent.rabaud@gmail.com> - 5.0.1-2
- Autogenerated by Bloom

* Tue Mar 26 2024 Vincent Rabaud <vincent.rabaud@gmail.com> - 5.0.1-1
- Autogenerated by Bloom

* Wed Mar 06 2024 Vincent Rabaud <vincent.rabaud@gmail.com> - 5.0.0-2
- Autogenerated by Bloom

