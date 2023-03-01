import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite
from robosuite.controllers import load_controller_config

# env = suite.make(
        # env_name="Lift",
        # robots="RS007N",
        # has_renderer=False,
        # has_offscreen_renderer=True,
        # use_camera_obs=True,
# )

controller_config = load_controller_config(default_controller="OSC_POSE")
"""
env = suite.make(
    "Lift",
    robots="RS007N",                        # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # each arm is controlled using OSC
    env_configuration="single-arm-opposed", # (two-arm envs only) arms face each other
    has_renderer=False,                     # no on-screen rendering
    has_offscreen_renderer=True,            # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=True,                    # provide image observations to agent
    camera_names="agentview",               # use "agentview" camera for observations
    camera_heights=480,                     # image height
    camera_widths=640,                      # image width
    camera_depths = True,                   # depth image 
    reward_shaping=True,                    # use a dense reward signal for learning
)
"""
env = suite.make(
    "Lift",
    robots="RS007N",                        # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # each arm is controlled using OSC
    env_configuration="single-arm-opposed", # (two-arm envs only) arms face each other
    has_renderer=True,                      # no on-screen rendering
    has_offscreen_renderer=False,           # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=1000,                           # each episode terminates after 200 steps
    use_object_obs=True,                    # don't provide object observations to agent
    use_camera_obs=False,                   # provide image observations to agent
    camera_names="agentview",               # use "agentview" camera for observations
    camera_heights=480,                     # image height
    camera_widths=640,                      # image width
    camera_depths = False,                  # depth image 
    reward_shaping=True,                    # use a dense reward signal for learning
)


obs = env.reset()

for _ in range(10000):
    action = np.random.randn(env.robots[0].dof)
    obs, reward, done, info = env.step(action)
    # if i % 20 == 0:
        # plt.imsave('test_color.png', obs['agentview_color'])
    env.render()
    
