# %% [markdown]
# ### [Optional] Download collected dataset

# %%
'''
If you want to use the collected dataset, please download it from Hugging Face.
'''
# !git clone https://huggingface.co/datasets/Jeongeun/omy_pnp_language

# %% [markdown]
# # Visualize your data
# 
# <img src="./media/data_v2.gif" width="480" height="360">
# 
# Visualize your action based on the reconstructed simulation scene. 
# 
# The main simulation is replaying the action.
# 
# The overlayed images on the top right and bottom right are from the dataset. 

# %%
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
ROOT = "./demo_data_language" # The root directory to save the demonstrations 
# If you have downloaded the dataset from Hugging Face, you can set the root to the directory where the dataset is stored
# ROOT = './omy_pnp_language' # if you want to use the example data provided, root = './omy_pnp_language' instead!
dataset = LeRobotDataset('omy_pnp_language', root=ROOT) # if youu want to use the example data provided, root = './omy_pnp_language' instead!

# If you want to use the collected dataset, please download it from Hugging Face.
# dataset = LeRobotDataset('omy_pnp_language', root='omy_pnp_language')

# %% [markdown]
# ## Load Dataset

# %%
import torch

class EpisodeSampler(torch.utils.data.Sampler):
    """
    Sampler for a single episode
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

# %%
# Select an episode index that you want to visualize
episode_index = 0

episode_sampler = EpisodeSampler(dataset, episode_index)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=1,
    batch_size=1,
    sampler=episode_sampler,
)


# %% [markdown]
# ## Visualize your Dataset on Simulation

# %%
from mujoco_env.y_env2 import SimpleEnv2
xml_path = './asset/example_scene_y2.xml'
PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')

# %%
step = 0
iter_dataloader = iter(dataloader)
PnPEnv.reset()

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Get the action from dataset
        data = next(iter_dataloader)
        if step == 0:
            # Reset the object pose based on the dataset
            instruction = data['task'][0]
            PnPEnv.set_instruction(instruction)
            PnPEnv.set_obj_pose(data['obj_init'][0,:3], data['obj_init'][0,3:6], data['obj_init'][0,6:9])
        # Get the action from dataset
        action = data['action'].numpy()
        obs = PnPEnv.step(action[0])

        # Visualize the image from dataset to rgb_overlay
        PnPEnv.rgb_agent = data['observation.image'][0].numpy()*255
        PnPEnv.rgb_ego = data['observation.wrist_image'][0].numpy()*255
        PnPEnv.rgb_agent = PnPEnv.rgb_agent.astype(np.uint8)
        PnPEnv.rgb_ego = PnPEnv.rgb_ego.astype(np.uint8)
        # 3 256 256 -> 256 256 3
        PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1,2,0))
        PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1,2,0))
        PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
        PnPEnv.render()
        step += 1

        if step == len(episode_sampler):
            # start from the beginning
            iter_dataloader = iter(dataloader)
            PnPEnv.reset()
            step = 0
    # PnPEnv

# %%
PnPEnv.env.close_viewer()

# %%
dataset.push_to_hub(upload_large_folder=True)

# %%



