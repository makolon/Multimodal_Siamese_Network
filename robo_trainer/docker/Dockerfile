FROM nvidia/cudagl:11.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y lsb-release
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-get install -y build-essential
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    g++-7 gcc-7 \
    software-properties-common \
    build-essential \
    wget \
    vim

RUN apt-get install -y python3-pip \
    && pip3 install catkin_tools \
    && rosdep init

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install --no-install-recommends -y\
                        wget \
                        git \
                        ros-noetic-tf* \
                        libopencv-dev \
                        libgstreamer*-dev \
                        intltool \
                        automake \
                        nodejs \
                        python3-catkin-pkg \
                        python3-catkin-tools \
                        python3-catkin-pkg-modules \
                        python3-rospkg-modules \
                        python3-opencv \
                        python3-empy \
                        python3-pip \
                        python3-dev \
                        build-essential \
                        openssh-client \
                        ros-noetic-rviz \
                        ros-noetic-rviz-visual-tools \
                        ros-noetic-combined-robot-hw \
                        ros-noetic-graph-msgs \
                        ros-noetic-moveit-ros-planning \
                        ros-noetic-moveit-ros-planning-interface \
                        ros-noetic-joint-state-publisher-gui \
                        ros-noetic-robot-state-publisher \
                        ros-noetic-ros-control \
                        ros-noetic-ros-controllers \
                        ros-noetic-gazebo-ros-control \
                        ros-noetic-moveit \
                        ros-noetic-moveit-commander \
                        ros-noetic-moveit-resources-prbt-moveit-config \
                        ros-noetic-pilz-industrial-motion-planner \
                        ros-noetic-diagnostic-updater \
                        ros-noetic-pcl-ros \
			ros-noetic-joy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
# create workspace and build cv_bridge for python3
RUN mkdir -p /root/catkin_ws/src \
    && cd catkin_ws \
    && catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
    && catkin config --install \
    && cd /root/catkin_ws/src \
    && git clone -b noetic https://github.com/ros-perception/vision_opencv.git \
    && git clone -b noetic-devel https://github.com/ros/geometry2.git \
    # khi rs007n
    && git clone https://github.com/Kawasaki-Robotics/khi_robot.git \
    && git clone https://github.com/Kawasaki-Robotics/khi_robot-release.git \
    && git clone https://github.com/Kawasaki-Robotics/khi_robot_app.git \
    ## moveit
    && git clone https://github.com/roboticsgroup/roboticsgroup_gazebo_plugins.git \
    && git clone https://github.com/ros-planning/moveit_visual_tools.git  \
    && git clone https://github.com/ros/urdf_tutorial.git \
    && git clone https://github.com/ros-planning/moveit_calibration.git \
    && git clone https://github.com/PickNikRobotics/rviz_visual_tools.git \
    && cd /root/catkin_ws \
    && source /opt/ros/noetic/setup.bash \
    && catkin build \
    && source install/setup.bash --extend

# rovi
RUN cd /root/catkin_ws/src \
    && git clone -b noetic https://github.com/YOODS/rovi.git \
    && git clone -b noetic-devel https://github.com/YOODS/rovi_utils.git \
    && cd /root/catkin_ws/ \
    && rm -rf devel/ build/ \
    && cd /root/catkin_ws/src/rovi \
    && bash -i ./Install.sh

# gelsight
RUN cd /root/catkin_ws/src \
    && git clone https://github.com/wjwwood/serial.git \
    && cd serial \
    && make \
    && make test \
    && make install 

RUN echo source /root/catkin_ws/devel/setup.bash >> .bashrc

ENV ROS_PACKAGE_PATH=/root/catkin_ws:$ROS_PACKAGE_PATH

ENV ROS_WORKSPACE=/root/catkin_ws

RUN source /opt/ros/noetic/setup.bash
###########

RUN apt-get update -y
RUN apt-get install -y curl ffmpeg
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN pip3 install opencv-python-headless
RUN pip3 install scikit-video
RUN pip --default-timeout=1000 install --ignore-installed open3d

RUN apt-get install -y git

RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install -y ninja-build
RUN pip3 install ninja pypng

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-7
RUN update-alternatives --config gcc


WORKDIR /
RUN git clone https://github.com/eigenteam/eigen-git-mirror.git && \
    cd eigen-git-mirror && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install

RUN touch a.txt

WORKDIR /
# RUN git clone https://github.com/CSAILVision/semantic-segmentation-pytorch.git && \
    # cd semantic-segmentation-pytorch && \
    # git checkout 5c2e9f6f3a231ae9ea150a0019d161fe2896efcf && \
    # mkdir ade20k-resnet50dilated-ppm_deepsup && cd ade20k-resnet50dilated-ppm_deepsup && \
    # wget -c http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth && \
    # wget -c http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth


RUN pip3 install pybullet

RUN apt-get update && apt-get install -y xvfb

RUN apt-get install -y xorg-dev \
    libglu1-mesa-dev \
    libsdl2-dev \
    libc++-7-dev \
    libc++abi-7-dev \
    ninja-build \
    libxi-dev \
    libtbb-dev \
    libosmesa6-dev

RUN apt -y autoremove
RUN apt-get update && apt-get -y upgrade

RUN apt-get -y install qtbase5-dev qttools5-dev-tools qt5-default

# RUN echo 'source /root/catkin_ws/install/setup.bash' >> ~/.bashrc
RUN echo 'export DISPLAY=:1' >> ~/.bashrc

RUN echo 'export HOME=/home/' >> ~/.bashrc
# RUN echo '/root/catkin_ws/devel/setup.bash' >> ~/.bashrc

WORKDIR /home

