# Multimodal-Siamese-Network
This repository contains an implementation of Multimodal Object Identification using Siamese Network created in my graduation project.

## Getting Started
We assume that you have access to a GPU with CUDA >= 11.0 support.
### Installation
1. Clone the repository
```
git clone https://github.com/makolon/Multimodal_Siamese_Network.git
```
2. Build docker image
#### For robot control
```
cd ./robo_trainer/docker/
./build.sh
```
#### For object identification
```
cd ./docker
./build.sh
```

3. Run docker container
#### For robot control
```
./robo_trainer/docker/
.run.sh
```

#### For object identification
```
cd ./docker
./run.sh
```

## Usage
#### For robot control
```
cd Multimoda_Siamese_Network/robo_trainer/catkin_ws
source devel/setup.bash
roslaunch rs007n/rs007n_default.launch
```

#### For object identification
```
cd Multimoda_Siamese_Network/siamese_network/
python3 train.py <params>
```

