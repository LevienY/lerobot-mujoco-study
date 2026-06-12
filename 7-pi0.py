# %% [markdown]
# # Deploy Trained pi_0 Policy
# 
# <img src="./media/rollout2.gif" width="480" height="360">
# 
# Deploy trained policy in simulation.

# %%
# !pip install pytest
# !pip install transformers==4.50.3

# %% [markdown]
# # Train pi_0 and Deploy

# %% [markdown]
# ### [Optional] Download Dataset

# %%
'''
If you want to use the collected dataset, please download it from Hugging Face.
'''
# !git clone https://huggingface.co/datasets/Jeongeun/omy_pnp_language

# %% [markdown]
# ## Step 1. Change the configuration fiel, pi0_omy.yaml

# %% [markdown]
# pi0_omy.yaml file
# ```
# dataset:
#   repo_id: omy_pnp
#   root: ./omy_pnp
# policy:
#   type : pi0
#   chunk_size: 5
#   n_action_steps: 5
# save_checkpoint: true
# output_dir: ./ckpt/pi0_omy
# batch_size: 16
# job_name : pi0_omy
# resume: false
# seed : 42
# num_workers: 8
# steps: 20_000
# eval_freq: -1 # No evaluation
# log_freq: 50
# save_checkpoint: true
# save_freq: 5_000
# use_policy_training_preset: true
#   
# wandb:
#   enable: true
#   project: pi0_omy
#   entity: <YOUR ENTITY for wandb>
#   disable_artifact: true
# ```

# %% [markdown]
# ## Step 2. Train Model.
# The code is tested on A100 

# %%
# !python train_model.py --config_path pi0_omy.yaml

# %% [markdown]
# ## Step 3. Deploy

# %%
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
import torch
from PIL import Image
import torchvision

# %% [markdown]
# ### Load Policy

# %%
device = 'cuda'

# %%
try:
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./demo_data_language')
except:
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./omy_pnp_language')
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
# Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
# we'll just use the defaults and so no arguments other than input/output features need to be passed.
# Temporal ensemble to make smoother trajectory predictions
cfg = PI0Config(input_features=input_features, output_features=output_features, chunk_size= 5, n_action_steps=5)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

# %%
# We can now instantiate our policy with this config and the dataset stats.
policy = PI0Policy.from_pretrained('./ckpt/pi0_omy/checkpoints/last/pretrained_model', dataset_stats=dataset_metadata.stats)
# You can load the trained policy from hub if you don't have the resources to train it.
# policy = PI0Policy.from_pretrained("Jeongeun/omy_pnp_pi0", config=cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)



# %% [markdown]
# ### Delopy Policy

# %%
from mujoco_env.y_env2 import SimpleEnv2
xml_path = './asset/example_scene_y3.xml'
PnPEnv = SimpleEnv2(action_type='joint_angle')

# %%
from torchvision import transforms
# Approach 1: Using torchvision.transforms
def get_default_transform(image_size: int = 224):
    """
    Returns a torchvision transform that:
     Converts to a FloatTensor and scales pixel values [0,255] -> [0.0,1.0]
    """
    return transforms.Compose([
        transforms.ToTensor(),  # PIL [0–255] -> FloatTensor [0.0–1.0], shape C×H×W
    ])

# %%
step = 0
PnPEnv.reset(seed=0)
policy.reset()
policy.eval()
save_image = True
IMG_TRANSFORM = get_default_transform()
while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Check if the task is completed
        success = PnPEnv.check_success()
        if success:
            print('Success')
            # Reset the environment and action queue
            policy.reset()
            PnPEnv.reset()
            step = 0
            save_image = False
        # Get the current state of the environment
        state = PnPEnv.get_joint_state()[:6]
        # Get the current image from the environment
        image, wirst_image = PnPEnv.grab_image()
        image = Image.fromarray(image)
        image = image.resize((256, 256))
        image = IMG_TRANSFORM(image)
        wrist_image = Image.fromarray(wirst_image)
        wrist_image = wrist_image.resize((256, 256))
        wrist_image = IMG_TRANSFORM(wrist_image)
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': [PnPEnv.instruction],
        }
        # Select an action
        action = policy.select_action(data)
        action = action[0,:7].cpu().detach().numpy()
        # Take a step in the environment
        _ = PnPEnv.step(action)
        PnPEnv.render()
        step += 1
        success = PnPEnv.check_success()
        if success:
            print('Success')
            break

# %%
# policy.push_to_hub(
#     repo_id='Jeongeun/omy_pnp_pi0',
#     commit_message='Add trained policy for PnP task',
# )

# %%



