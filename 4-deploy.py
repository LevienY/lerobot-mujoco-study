# %% [markdown]
# # Deploy Trained Policy
# 
# <img src="./media/rollout.gif" width="480" height="360">
# 
# Deploy trained policy in simulation.

# %%
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
import torch
from PIL import Image
import torchvision

# %% [markdown]
# ## Load Policy

# %%
device = 'cuda'

# %%
dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root='./demo_data')
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
input_features.pop("observation.wrist_image")
# Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
# we'll just use the defaults and so no arguments other than input/output features need to be passed.
# Temporal ensemble to make smoother trajectory predictions
cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size= 10, n_action_steps=1, temporal_ensemble_coeff = 0.9)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
# We can now instantiate our policy with this config and the dataset stats.
policy = ACTPolicy.from_pretrained('./ckpt/act_y', config = cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)

# %% [markdown]
# ## Load Environment

# %%
from mujoco_env.y_env import SimpleEnv
xml_path = './asset/example_scene_y.xml'
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

# %% [markdown]
# ## Roll-Out Your Policy

# %%
step = 0
PnPEnv.reset(seed=0)
policy.reset()
policy.eval()
save_image = True
img_transform = torchvision.transforms.ToTensor()
while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Check if the task is completed
        success = PnPEnv.check_success()
        if success:
            print('Success')
            # Reset the environment and action queue
            policy.reset()
            PnPEnv.reset(seed=0)
            step = 0
            save_image = False
        # Get the current state of the environment
        state = PnPEnv.get_ee_pose()
        # Get the current image from the environment
        image, wirst_image = PnPEnv.grab_image()
        image = Image.fromarray(image)
        image = image.resize((256, 256))
        image = img_transform(image)
        wrist_image = Image.fromarray(wirst_image)
        wrist_image = wrist_image.resize((256, 256))
        wrist_image = img_transform(wrist_image)
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': ['Put mug cup on the plate'],
            'timestamp': torch.tensor([step/20]).to(device)
        }
        # Select an action
        action = policy.select_action(data)
        action = action[0].cpu().detach().numpy()
        # Take a step in the environment
        _ = PnPEnv.step(action)
        PnPEnv.render()
        step += 1
        success = PnPEnv.check_success()
        if success:
            print('Success')
            break

# %%


# %%



