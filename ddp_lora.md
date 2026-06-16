下面是每个参数的含义和设置建议：

1. CUDA_VISIBLE_DEVICES=0,1,2,3
含义：告诉当前程序只能看见哪些 GPU。

服务器上可能有 4 张以上的卡，但这里只让程序使用编号 0、1、2、3 的四张卡。
如果不指定，程序默认使用所有可见 GPU（可能导致与其他任务冲突）。
设置方法：

用逗号分隔 GPU 序号，如 0,1,2,3
如果只希望用第 2、3 张卡：CUDA_VISIBLE_DEVICES=2,3
也可以用 CUDA_VISIBLE_DEVICES=（空值）来隐藏所有 GPU（仅 CPU 测试）


2. LORA_RANK=32 LORA_ALPHA=64 LORA_DROPOUT=0.05
这三个是我们自定义的环境变量，用于控制 LoRA 微调的参数。

注意：你的 
train_model_ddp.py
 里需要包含前面我给的 LoRA 注入代码才会读取这些环境变量。

变量	含义	常用取值范围
LORA_RANK	LoRA 矩阵的秩（低秩分解的维度），决定可训练参数数量	8, 16, 32, 64（越大可训练参数越多，拟合能力更强）
LORA_ALPHA	缩放因子，实际学习率下 LoRA 权重为 (alpha / rank) * ΔW	通常设为 rank 的 1~2 倍，例如 rank=32, alpha=64
LORA_DROPOUT	LoRA dropout 概率，用于防止过拟合	0.0 ~ 0.1（数据集小时可提高一点）
如何选择？

数据量少（几千条轨迹）：rank=16, alpha=32 即可，防止过拟合
数据量中等：rank=32, alpha=64 比较平衡
数据量大：rank=64, alpha=128 可以提供更强拟合能力
如果不确定，用默认值 rank=32, alpha=64, dropout=0.0
3. torchrun --nproc_per_node=4
这是 PyTorch 分布式启动工具。

参数	含义
torchrun	PyTorch 提供的多进程启动器，用于 DDP 训练
--nproc_per_node=4	单台机器上启动的进程数，每个进程对应一张 GPU，这里启动 4 个进程，分别使用 GPU 0/1/2/3
其他常用参数（可选）：

--master_port=29500：指定通信端口（默认 29500，如果端口冲突可以改）
--nnodes=1：节点数（单机训练不用改）
4. 
train_model_ddp.py --config_path pi0_omy.yaml
train_model_ddp.py
：你的训练脚本（已修改为支持 DDP 和 LoRA）
--config_path pi0_omy.yaml：传给训练脚本的配置文件路径，里面定义了数据集路径、策略类型、训练步数等




100 条数据，LoRA 秩要降低 + dropout 要加大：

CUDA_VISIBLE_DEVICES=0,1,2,3 \
LORA_RANK=8 LORA_ALPHA=16 LORA_DROPOUT=0.1 \
torchrun --nproc_per_node=4 \
    train_model_ddp.py \
    --config_path pi0_omy.yaml

4 卡 DDP 下，每个 epoch 的步数：
steps_per_epoch = 数据集大小 / (per_gpu_batch_size × GPU数)
                = 100 / (4 × 4)
                = 100 / 16
                ≈ 6.25 步/epoch

total_epochs = steps / steps_per_epoch
             = 2000 / 6.25
             ≈ 320 epochs


训练时观察 wandb 上的 loss 曲线：

正常：loss 缓慢下降，趋于平稳
过拟合：loss 快速降到接近 0，说明模型在背数据
如果发现过拟合，可以：

进一步降低 LORA_RANK=4
增加 LORA_DROPOUT=0.2
减少 steps=1000


CUDA_VISIBLE_DEVICES=0,1,2,3 \
LORA_RANK=8 LORA_ALPHA=16 LORA_DROPOUT=0.1 \
torchrun --nproc_per_node=4 --master_port=29600 \
    train_model_ddp.py \
    --config_path pi0_omy.yaml



nvidia-smi | grep '.venv/bin/python' | awk '{print $5}' | xargs -r kill -9






你混淆了一个概念：LoRA 减少的是可训练参数量（梯度/优化器状态），不减少前向传播的显存。前向传播时整个 3.5B 模型的所有层都要计算激活值，每个 GPU 上有完整的模型副本。

你的 4 张 4090（24G），3.5B 模型在 bf16 下权重约 7G，加上前向的激活值（取决于 batch_size 和序列长度），非常容易超出 24G。

解决办法（按优先级排列）
1. 降低 batch_size （最直接）
当前代码中 cfg.batch_size 是多少？如果是 8 或 4，先改成 1：


Apply
# 启动命令里加：
--batch_size 1
2. 启用 gradient checkpointing（用计算换显存）
Pi0 的 Paligemma 底层支持 gradient checkpointing。在 make_policy 之后加上：


Apply
# 在 make_policy 之后，DDP 包装之前
if hasattr(policy, "model") and hasattr(policy.model, "config"):
    policy.model.gradient_checkpointing_enable()
或者在 cfg 的 policy 配置里加上对应的选项（如果支持的话）。

3. 换小模型 smolvla
SmolVLA 比 Pi0 小很多（~300M vs 3.5B），显存友好得多。你代码里已经写了 smolvla 的分支，直接用：

Run
# 启动时设置 policy type 为 smolvla
--policy.type smolvla
4. 只用 1-2 张卡，不要 4 卡 DDP
4 张卡 DDP 对显存不会减少，反而每张卡都要一个完整副本。如果 1 张卡 batch_size=1 勉强能跑，4 张卡只是并行计算，显存压力一样。所以要么 1 卡 + batch_size=1，要么换模型。

5. 显存碎片优化
从日志里已经提示了：

try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

启动命令前加上：