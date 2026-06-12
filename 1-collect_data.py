# %% [markdown]
# # Collect Demonstration from Keyboard
# 
# Collect demonstration data for the given environment.
# The task is to pick a mug and place it on the plate. The environment recognizes the success if the mug is on the plate, gthe ripper opened, and the end-effector positioned above the mug.
# 
# <img src="./media/teleop.gif" width="480" height="360">
# 
# Use WASD for the xy plane, RF for the z-axis, QE for tilt, and ARROWs for the rest of rthe otations. 
# 
# SPACEBAR will change your gripper's state, and Z key will reset your environment with discarding the current episode data.
# 
# For overlayed images, 
# - Top Right: Agent View 
# - Bottom Right: Egocentric View
# - Top Left: Left Side View
# - Bottom Left: Top View
# test
# %%
import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.y_env import SimpleEnv
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# %%
# If you want to randomize the object positions, set this to None
# If you fix the seed, the object positions will be the same every time
# SEED = 0 
SEED = None #<- Uncomment this line to randomize the object positions

REPO_NAME = 'omy_pnp'
NUM_DEMO = 1 # Number of demonstrations to collect
ROOT = "./demo_data" # The root directory to save the demonstrations

# %%
TASK_NAME = 'Put mug cup on the plate' 
xml_path = './asset/example_scene_y.xml'
# Define the environment
PnPEnv = SimpleEnv(xml_path, seed = SEED, state_type = 'joint_angle')

# %% [markdown]
# ## Define Dataset Fatures and Create your dataset!
# The dataset is contained as follows:
# ```
# fps = 20,
# features={
#     "observation.image": {
#         "dtype": "image",
#         "shape": (256, 256, 3),
#         "names": ["height", "width", "channels"],
#     },
#     "observation.wrist_image": {
#         "dtype": "image",
#         "shape": (256, 256, 3),
#         "names": ["height", "width", "channel"],
#     },
#     "observation.state": {
#         "dtype": "float32",
#         "shape": (6,),
#         "names": ["state"], # x, y, z, roll, pitch, yaw
#     },
#     "action": {
#         "dtype": "float32",
#         "shape": (7,),
#         "names": ["action"], # 6 joint angles and 1 gripper
#     },
#     "obj_init": {
#         "dtype": "float32",
#         "shape": (6,),
#         "names": ["obj_init"], # just the initial position of the object. Not used in training.
#     },
# },
# ```
# 
# 
# This will make the dataset on './demo_data' folder, which will look like this,
# 
# ```
# .
# ├── data
# │   ├── chunk-000
# │   │   ├── episode_000000.parquet
# │   │   └── ...
# ├── meta
# │   ├── episodes.jsonl
# │   ├── info.json
# │   ├── stats.json
# │   └── tasks.jsonl
# └── 
# ```
# 

# %%
create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False


if create_new:
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root = ROOT, 
                robot_type="omy",
                fps=20, # 20 frames per second
                features={
                    "observation.image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["state"], # x, y, z, roll, pitch, yaw
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["action"], # 6 joint angles and 1 gripper
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["obj_init"], # just the initial position of the object. Not used in training.
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,
        )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

# %% [markdown]
# ## Keyboard Control
# You can teleop your robot with keyboard and collect dataset
# ```
# ---------     -----------------------
#    w       ->        backward
# s  a  d        left   forward   right
# ---------      -----------------------
# In x, y plane
# 
# ---------
# R: Moving Up
# F: Moving Down
# ---------
# In z axis
# 
# ---------
# Q: Tilt left
# E: Tilt right
# UP: Look Upward
# Down: Look Donward
# Right: Turn right
# Left: Turn left
# ---------
# For rotation
# 
# ---------
# SPACEBAR: Toggle Gripper
# --------
# 
# ---------
# z: reset
# --------
# ```
# Reseting your environment will remove the cache data of the current demonstration and restart collection.

# %% [markdown]
# ### Now let's teleop our robot and collect data!
# 
# **To receive the success signal, you have to release the gripper and move upwards above the mug!**

# %%
action = np.zeros(7)
episode_id = 0
record_flag = False # Start recording when the robot starts moving
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # check if the episode is done
        done = PnPEnv.check_success()
        if done: 
            # Save the episode data and reset the environment
            dataset.save_episode()
            PnPEnv.reset(seed = SEED)
            episode_id += 1
        # Teleoperate the robot and get delta end-effector pose with gripper
        action, reset  = PnPEnv.teleop_robot()
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("Start recording")
        if reset:
            # Reset the environment and clear the episode buffer
            # This can be done by pressing 'z' key
            PnPEnv.reset(seed=SEED)
            # PnPEnv.reset()
            dataset.clear_episode_buffer()
            record_flag = False
        # Step the environment
        # Get the end-effector pose and images
        ee_pose = PnPEnv.get_ee_pose()
        agent_image,wrist_image = PnPEnv.grab_image()
        # # resize to 256x256
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        joint_q = PnPEnv.step(action)
        if record_flag:
            # Add the frame to the dataset
            dataset.add_frame( {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose, 
                    "action": joint_q,
                    "obj_init": PnPEnv.obj_init_pose,
                    # "task": TASK_NAME,
                }, task = TASK_NAME
            )
        PnPEnv.render(teleop=True)

# %%
PnPEnv.env.close_viewer()

# %%
# Clean up the images folder
import shutil
shutil.rmtree(dataset.root / 'images')

# %%



