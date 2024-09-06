# Original LoRA train script by @Akegarasu ; rewritten in Python by LJRE.
import subprocess
import os
import folder_paths
import random
from comfy import model_management
import torch
from .image_helpers import ImageHelpers

#Train data path | 设置训练用模型、图片
#pretrained_model = "E:\AI-Image\ComfyUI_windows_portable_nvidia_cu121_or_cpu\ComfyUI_windows_portable\ComfyUI\models\checkpoints\MyAnimeModel.ckpt"
is_v2_model = 0  # SD2.0 model | SD2.0模型 2.0模型下 clip_skip 默认无效
parameterization = 0  # parameterization | 参数化 本参数需要和 V2 参数同步使用 实验性功能
#train_data_dir = "" # train dataset path | 训练数据集路径
reg_data_dir = ""  # directory for regularization images | 正则化数据集路径，默认不使用正则化图像。

# Network settings | 网络设置
network_module = "networks.lora"  # 网络设置为LoRA训练
network_weights = ""  # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。
network_dim = 128  # updated based on JSON
network_alpha = 128  # updated based on JSON

# Train related params | 训练相关参数
resolution = "512,512"  # image resolution w,h. 图片分辨率，宽,高。支持非正方形，但必须是 64 倍数。
#batch_size = 1 # batch size | batch 大小
#max_train_epoches = 10 # max train epoches | 最大训练 epoch
#save_every_n_epochs = 10 # save every n epochs | 每 N 个 epoch 保存一次

train_unet_only = 0  # train U-Net only | 仅训练 U-Net
train_text_encoder_only = 0  # train Text Encoder only | 仅训练 文本编码器
stop_text_encoder_training = 0  # stop text encoder training | 停止训练文本编码器

noise_offset = 0  # noise offset | 添加噪声偏移
keep_tokens = 0  # 保留前 N 个 token
min_snr_gamma = 0  # 最小 SNR 值

# Learning rate | 学习率
lr = "1e-4"  # learning rate | 学习率
unet_lr = "1e-4"  # U-Net learning rate | U-Net 学习率
text_encoder_lr = "1e-5"  # Text Encoder learning rate | 文本编码器 学习率
lr_scheduler = "constant"  # updated based on JSON
lr_warmup_steps = 0  # warmup steps | 学习率预热步数
lr_restart_cycles = 1  # 余弦退火重启次数

# Optimizer settings | 优化器设置
optimizer_type = "AdamW8bit"  # Optimizer type | AdamW8bit 优化器

# Output settings | 输出设置
#output_name = "Pkmn3GTest" # output model name | 模型保存名称
save_model_as = "safetensors"  # 模型保存格式

# Resume training state | 恢复训练设置
save_state = 0  # save training state | 保存训练状态
resume = ""  # resume from state | 恢复训练

# Other settings | 其他设置
min_bucket_reso = 256  # arb 最小分辨率
max_bucket_reso = 2048  # arb 最大分辨率
persistent_data_loader_workers = 0  # updated based on JSON
#clip_skip = 2 # clip skip
multi_gpu = 0  # multi GPU | 多显卡训练
lowram = 0  # 低内存模式

# LyCORIS training settings | LyCORIS 训练设置
algo = "lora"  # LyCORIS 网络算法
conv_dim = 1  # updated based on JSON
conv_alpha = 1  # updated based on JSON
dropout = "0"  # dropout 概率

# Remote logging settings | 远程记录设置
use_wandb = 0  # disable wandb logging

# C:\Users\MSI-Win11\Downloads\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\mengy_training
image_sd_path = "ImageTraining"
image_path = "100_TrainingImgs"
image_full_path = os.path.join(image_sd_path, image_path)

output_dir = os.path.abspath("SafetensorOutput")
logging_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
log_prefix = ''
mixed_precision = 'fp16'
caption_extension = '.txt'


os.environ['HF_HOME'] = "huggingface"
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = "1"
ext_args = []
launch_args = []

def GetTrainScript(script_name:str):
    # Current file directory from __file__
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    sd_script_dir = os.path.join(current_file_dir, "sd-scripts")
    train_script_path = os.path.join(sd_script_dir, f"{script_name}.py")
    return train_script_path, sd_script_dir

class LoraTraininginComfyAdvanced:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "model_type": (["sd1.5", "sd2.0", "sdxl"], ),
                "networkmodule": (["networks.lora", "lycoris.kohya"], ),
                "networkdimension": ("INT", {"default": 128, "min":0}),
                "networkalpha": ("INT", {"default":128, "min":0}),
                "resolution_width": ("INT", {"default":512, "step":64}),
                "resolution_height": ("INT", {"default":512, "step":64}),
                "batch_size": ("INT", {"default": 2, "min":1}),
                "max_train_epoches": ("INT", {"default":1, "min":1}),
                "save_every_n_epochs": ("INT", {"default":1, "min":1}),
                "keeptokens": ("INT", {"default":0, "min":0}),
                "minSNRgamma": ("FLOAT", {"default":0, "min":0, "step":0.1}),
                "learningrateText": ("FLOAT", {"default":0.0001, "min":0, "step":0.00001}),
                "learningrateUnet": ("FLOAT", {"default":0.0001, "min":0, "step":0.00001}),
                "learningRateScheduler": (["cosine_with_restarts", "linear", "cosine", "polynomial", "constant", "constant_with_warmup"], {"default": "constant"}),
                "lrRestartCycles": ("INT", {"default":1, "min":1}),
                "optimizerType": (["AdamW8bit", "Lion8bit", "SGDNesterov8bit", "AdaFactor", "prodigy"], {"default": "AdamW8bit"}),
                "output_name": ("STRING", {"default":'Brookreator'}),
                "algorithm": (["lora","loha","lokr","ia3","dylora", "locon"], {"default": "lora"}),
                "networkDropout": ("FLOAT", {"default": 0, "step":0.1}),
                "clip_skip": ("INT", {"default":2, "min":1}),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "loratraining"

    OUTPUT_NODE = True

    CATEGORY = "LJRE/LORA"

    def loratraining(self, 
                     images, ckpt_name, model_type, networkmodule, 
                     networkdimension, networkalpha, 
                     resolution_width, resolution_height, batch_size, max_train_epoches, save_every_n_epochs, keeptokens, minSNRgamma, 
                     learningrateText, learningrateUnet, learningRateScheduler, lrRestartCycles, optimizerType, output_name, algorithm, networkDropout, clip_skip):
        #free memory first of all
        loadedmodels=model_management.current_loaded_models
        unloaded_model = False
        for i in range(len(loadedmodels) -1, -1, -1):
            m = loadedmodels.pop(i)
            m.model_unload()
            del m
            unloaded_model = True
        if unloaded_model:
            model_management.soft_empty_cache()

        
        h = ImageHelpers()
        h.clean_dir(image_full_path)
        h.save_images(images, image_full_path)
        data_path = os.path.abspath(image_sd_path)
            
        #print(model_management.current_loaded_models)
        #loadedmodel = model_management.LoadedModel()
        #loadedmodel.model_unload(self, current_loaded_models)
        
        #transform backslashes into slashes for user convenience.
        train_data_dir = data_path.replace( "\\", "/")
        if data_path == "Insert path of image folders":
            raise ValueError("Please insert the path of the image folders.")

        if output_name == 'Desired name for LoRA.': 
            raise ValueError("Please insert the desired name for LoRA.")
        
        #ADVANCED parameters initialization
        train_script_name = "train_network"
        
        if model_type == "sd1.5":
            ext_args.append(f"--clip_skip={clip_skip}")
        elif model_type == "sd2.0":
            ext_args.append("--v2")
        elif model_type == "sdxl":
            train_script_name = "sdxl_train_network"
        
        network_module = networkmodule
        network_dim = networkdimension
        network_alpha = networkalpha
        resolution = f"{resolution_width},{resolution_height}"
        
        formatted_value = str(format(learningrateText, "e")).rstrip('0').rstrip()
        text_encoder_lr = ''.join(c for c in formatted_value if not (c == '0'))
        
        formatted_value2 = str(format(learningrateUnet, "e")).rstrip('0').rstrip()
        unet_lr = ''.join(c for c in formatted_value2 if not (c == '0'))
        
        keep_tokens = keeptokens
        min_snr_gamma = minSNRgamma
        lr_scheduler = learningRateScheduler
        lr_restart_cycles = lrRestartCycles
        optimizer_type = optimizerType
        algo = algorithm
        dropout = f"{networkDropout}"

        #generates a random seed
        theseed = random.randint(0, 2^32-1)
        
        if multi_gpu:
            launch_args.append("--multi_gpu")
        
        if network_module == "lycoris.kohya":
            ext_args.extend([
                f"--network_args",
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}",
                f"algo={algo}",
                f"dropout={dropout}"
            ])

        if lowram:
            ext_args.append("--lowram")

        if parameterization:
            ext_args.append("--v_parameterization")

        if train_unet_only:
            ext_args.append("--network_train_unet_only")

        if train_text_encoder_only:
            ext_args.append("--network_train_text_encoder_only")

        if network_weights:
            ext_args.append(f"--network_weights={network_weights}")

        # if reg_data_dir:
        #     ext_args.append(f"--reg_data_dir={reg_data_dir}")

        if optimizer_type:
            ext_args.append(f"--optimizer_type={optimizer_type}")

        if optimizer_type == "DAdaptation":
            ext_args.append("--optimizer_args")
            ext_args.append("decouple=True")

        if network_module == "lycoris.kohya":
            ext_args.extend([
                f"--network_args",
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}",
                f"algo={algo}",
                f"dropout={dropout}"
            ])

        if noise_offset != 0:
            ext_args.append(f"--noise_offset={noise_offset}")

        if stop_text_encoder_training != 0:
            ext_args.append(f"--stop_text_encoder_training={stop_text_encoder_training}")

        if save_state == 1:
            ext_args.append("--save_state")

        if resume:
            ext_args.append(f"--resume={resume}")

        if min_snr_gamma != 0:
            ext_args.append(f"--min_snr_gamma={min_snr_gamma}")

        if persistent_data_loader_workers:
            ext_args.append("--persistent_data_loader_workers")

        if use_wandb == 1:
            ext_args.append("--log_with=all")
            if wandb_api_key:
                ext_args.append(f"--wandb_api_key={wandb_api_key}")
            if log_tracker_name:
                ext_args.append(f"--log_tracker_name={log_tracker_name}")
        else:
            ext_args.append("--log_with=tensorboard")

        launchargs=' '.join(launch_args)
        extargs=' '.join(ext_args)

        pretrained_model = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        #Looking for the training script.
        nodespath, sd_script_dir = GetTrainScript(script_name=train_script_name)
        print(nodespath)
        print(sd_script_dir)

        command = (
            f"python -m accelerate.commands.launch {launchargs} "
            f'--num_cpu_threads_per_process=8 "{nodespath}" '
            f"--enable_bucket "
            f"--pretrained_model_name_or_path={pretrained_model} "
            f'--train_data_dir="{train_data_dir}" '
            f'--output_dir="{output_dir}" '
            f'--logging_dir="{logging_dir}" '
            f"--log_prefix={output_name} "
            f"--resolution={resolution} "
            f"--network_module={network_module} "
            f"--max_train_epochs={max_train_epoches} "
            f"--learning_rate={lr} "
            f"--unet_lr={unet_lr} "
            f"--text_encoder_lr={text_encoder_lr} "
            f"--lr_scheduler={lr_scheduler} "
            f"--lr_warmup_steps={lr_warmup_steps} "
            f"--lr_scheduler_num_cycles={lr_restart_cycles} "
            f"--network_dim={network_dim} "
            f"--network_alpha={network_alpha} "
            f"--output_name={output_name} "
            f"--train_batch_size={batch_size} "
            f"--save_every_n_epochs={save_every_n_epochs} "
            f'--mixed_precision="fp16" '
            f'--save_precision="fp16" '
            f"--seed={theseed} "
            f"--cache_latents "
            f"--prior_loss_weight=1 "
            f"--max_token_length=225 "
            f'--caption_extension=".txt" '
            f"--save_model_as={save_model_as} "
            f"--min_bucket_reso={min_bucket_reso} "
            f"--max_bucket_reso={max_bucket_reso} "
            f"--keep_tokens={keep_tokens} "
            f"--xformers "
            ## BROOK
            f"--bucket_no_upscale "
            ## END BROOK
            f"{extargs}"
        )
        
        print(command)
        subprocess.run(command, cwd=sd_script_dir)
        print("Train finished")
        #input()
        return ()