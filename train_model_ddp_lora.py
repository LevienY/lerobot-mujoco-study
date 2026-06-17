#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 在 train_model.py 顶部添加
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    """初始化多卡通信"""
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()




import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

import peft
# 在现有 import 后面添加
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()

    # 穿透 DDP 和 PEFT wrapper，拿到底层的 PI0Policy / SmolVLAPolicy
    inner = policy.module if hasattr(policy, 'module') else policy       # 穿透 DDP
    actual_model = inner.base_model.model if hasattr(inner, 'base_model') else inner  # 穿透 PEFT


    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = actual_model.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    # ====== DDP 初始化 ======
    if 'LOCAL_RANK' in os.environ:
        local_rank = setup_ddp()
        is_distributed = True
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {local_rank}] DDP initialized on GPU {local_rank}")
    else:
        local_rank = 0
        is_distributed = False
        device = get_safe_torch_device(cfg.policy.device, log=True)

    if not is_distributed:
        get_safe_torch_device(cfg.policy.device, log=True)  # 单卡时打印设备信息
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # ==========================

    if cfg.wandb.enable and cfg.wandb.project and local_rank == 0:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)
    

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # ====== 构建 DataLoader ======
    if hasattr(cfg.policy, "drop_n_last_frames"):
        # 原始 EpisodeAwareSampler 逻辑，但 DDP 下仍保留（做简单分片）
        # 注意：这里的 episode_data_index 需要是全局的，每个 rank 分到不同 episodes
        # 简单写法：让每个 rank 拿到其中的一部分（假设 episode 数量足够）
        from lerobot.common.datasets.sampler import EpisodeAwareSampler
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        if is_distributed:
            # 使用 DistributedSampler 包裹 EpisodeAwareSampler 的索引？
            # 更稳妥的做法：直接换成 DistributedSampler，放弃 episode-aware 特征
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=True
            )
    else:
        if is_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=True
            )
        else:
            sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False if sampler is not None else True,  # sampler 存在时 shuffle 必须为 False
        sampler=sampler,
        pin_memory=True,
        drop_last=True if is_distributed else False,    # DDP 时建议 drop_last 避免梯度不一致
    )
    dl_iter = cycle(dataloader)
    # ===============================

   

    # 训练循环

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # ====== 创建 Policy ======
    logging.info("Creating policy")
    if cfg.policy.type == "pi0":
        cfg.policy.pretrained_path = 'lerobot/pi0'
    elif cfg.policy.type == 'smolvla':
        cfg.policy.pretrained_path = 'lerobot/smolvla_base'

    if is_distributed:
        # 关键：先把 device 设为 cpu，防止 4 个进程同时往 GPU 0 加载
        original_device = cfg.policy.device
        cfg.policy.device = "cpu"

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    if is_distributed:
        cfg.policy.device = original_device  # 恢复原设置
        # 现在安全地把模型搬到当前 rank 对应的 GPU
        policy = policy.to(device)
    else:
        policy.to(device)
    # ========================

    # ====== 手动注入 LoRA（通过环境变量控制） ======
    lora_rank = int(os.environ.get('LORA_RANK', '0'))
    if lora_rank > 0 and HAS_PEFT:
        lora_alpha = int(os.environ.get('LORA_ALPHA', str(lora_rank)))
        lora_dropout = float(os.environ.get('LORA_DROPOUT', '0.0'))
        
        if local_rank == 0:
            logging.info(f"Applying LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
        
        # 自动查找 attention 相关的线性层作为 target_modules
        target_modules = []
        for name, module in policy.named_modules():
            if isinstance(module, torch.nn.Linear) and any(
                s in name for s in ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'query', 'value', 'key', 'out_proj']
            ):
                module_name = name.split('.')[-1] if '.' in name else name
                if module_name not in target_modules:
                    target_modules.append(module_name)
        
        # 如果没找到特定命名，fallback 到所有 Linear 层
        if not target_modules:
            if local_rank == 0:
                logging.warning("No attention modules found by name, applying LoRA to all Linear layers")
            target_modules = None  # peft 会自动选择
        
        if local_rank == 0:
            logging.info(f"LoRA target_modules: {target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules if target_modules else "all-linear",
        )
        try:
            policy = get_peft_model(policy, lora_config)
            if local_rank == 0:
                policy.print_trainable_parameters()
        except Exception as e:
            logging.error(f"LoRA injection failed: {e}")
            raise
    elif lora_rank > 0 and not HAS_PEFT:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")
    # ==============================================

    # ====== 创建优化器（必须在 DDP 包装之前） ======
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    # ============================================

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # ====== 用 DDP 包装 ======
    if is_distributed:
        policy = DDP(policy, device_ids=[local_rank], output_device=local_rank,
                     find_unused_parameters=False)   # LoRA 时建议 False
    # =========================

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if local_rank == 0:  # 仅在主进程打印
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step and local_rank == 0:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step and local_rank == 0:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

            # 获取真正要保存的模型（穿透 DDP）
            model_to_save = policy.module if is_distributed else policy

            # ====== LoRA 训练时，手动保存基础模型的 config.json ======
            if lora_rank > 0 and HAS_PEFT and isinstance(model_to_save, peft.PeftModel):
                # 从 PeftModel 中取出基础策略（PI0Policy / SmolVLAPolicy）
                base_model = model_to_save.base_model.model
                base_config = base_model.config  # PI0Config / SmolVLAConfig 对象
                # 转为字典并写入 config.json
                import json
                pretrained_dir = checkpoint_dir / "pretrained_model"
                with open(pretrained_dir / "config.json", "w") as f:
                    json.dump(base_config.to_dict(), f, indent=2)
                logging.info(f"Saved base model config.json to {pretrained_dir}")
            # ========================================================

            save_checkpoint(checkpoint_dir, step, cfg,
                            model_to_save, 
                            optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            if is_distributed:
                dist.barrier()   # 确保所有进程同时进入或等待
            
            if local_rank == 0:  # 只有 rank 0 执行评估
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    eval_info = eval_policy(
                        eval_env,
                        policy.module if is_distributed else policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

            if is_distributed:
                dist.barrier()  # 所有进程等待 rank 0 评估完成后，再一起继续训练
    if eval_env:
        eval_env.close()

    if is_distributed:
        cleanup_ddp()
    if local_rank == 0:
        logging.info("End of training")



if __name__ == "__main__":
    init_logging()
    train()