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

RUN apt-get install -y python3-pip
RUN pip3 install catkin_tools

ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV FORCE_CUDA="1"
ENV CUDA_VISIBLE_DEVICES="0"

RUN rosdep init

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install --no-install-recommends -y\
                       wget \
                       git \
                       ros-noetic-tf* \
                       libopencv-dev \
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
    && cd /root/catkin_ws \
    && source /opt/ros/noetic/setup.bash \
    # && catkin build cv_ridge \
    # && catkin build geometry2 \
    && catkin build \
    && source install/setup.bash --extend

RUN echo source /root/catkin_ws/devel/setup.bash >> .bashrc

ENV ROS_PACKAGE_PATH=/root/catkin_ws:$ROS_PACKAGE_PATH

ENV ROS_WORKSPACE=/root/catkin_ws

RUN source /opt/ros/noetic/setup.bash

###########

RUN apt-get update -y
RUN apt-get install -y curl ffmpeg
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN pip3 install opencv-python
RUN apt-get install -y python3-opencv
RUN apt-get install -y git

RUN pip3 install torch==1.7.1 torchvision==0.8.2
RUN pip3 install scipy
RUN pip3 install gym

ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

RUN apt-get install -y ninja-build
RUN pip3 install ninja pypng

#RUN nvcc -v
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

RUN pip3 install pybullet

RUN apt-get update && apt-get install -y xvfb

RUN apt-get install -y xorg-dev \
    libglu1-mesa-dev \
    # Filament build-from-source deps
    libsdl2-dev \
    libc++-7-dev \
    libc++abi-7-dev \
    ninja-build \
    libxi-dev \
    # ML deps
    libtbb-dev \
    # Headless rendering deps
    libosmesa6-dev

RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
RUN apt-get update && apt-get install -y cmake
RUN apt -y autoremove

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
        wandb \
        pytransform3d

RUN echo 'source /root/catkin_ws/install/setup.bash' >> ~/.bashrc
RUN echo 'export DISPLAY=:1' >> ~/.bashrc

WORKDIR /home
