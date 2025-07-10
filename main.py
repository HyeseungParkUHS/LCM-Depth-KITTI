import os
import re
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torchvision import transforms as T
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
# import pandas as pd
# import json
from torch.utils.data import DistributedSampler
import random
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
# from diffusers.models.attention_processor import AttnProcessor
from huggingface_hub import login, whoami
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
# from torchvision.utils import save_image
# import cv2
import torch.nn as nn
import copy
import torch.nn.functional as F
import torchvision.utils as vutils
# import pytorch_msssim
# import lpips
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from contextlib import nullcontext
# import OpenEXR, Imath
from torchvision import transforms
from glob import glob

# 기본 설정
width = 320
height = 320

# GPU 메모리 최적화 설정
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def huggingface_login(token):
    """Hugging Face 로그인"""
    login(token)
    user_info = whoami()
    print(f"🔐 Hugging Face 로그인 완료: {user_info['name']}")


def setup_ddp(rank, world_size):
    """DDP 설정"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"DDP Setup Complete for rank {rank}.")


def cleanup_ddp():
    """DDP 정리"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Error during DDP cleanup: {str(e)}")

def wrap_model_ddp(model, rank):
    return DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

class ProjectPaths:
    def __init__(self, base_dir=None):
        """프로젝트 디렉토리 구조 관리"""
        self.base_dir = base_dir

        # 주요 디렉토리 경로 설정
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        self.logs_dir = os.path.join(base_dir, "logs")
        self.samples_dir = os.path.join(base_dir, "samples")

        # 세부 디렉토리 경로 설정
        self.checkpoint_subdirs = {
            'latest': os.path.join(self.checkpoints_dir, "latest"),
            'best': os.path.join(self.checkpoints_dir, "best"),
            # 'periodic': os.path.join(self.checkpoints_dir, "periodic")
        }

        self.samples_subdirs = {
            'training': os.path.join(self.samples_dir, "training"),
        }

        # 모든 디렉토리 생성
        self._create_directories()

    def _create_directories(self):
        """모든 필요한 디렉토리 생성"""
        # 메인 디렉토리들
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.samples_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 체크포인트 서브디렉토리
        for subdir in self.checkpoint_subdirs.values():
            os.makedirs(subdir, exist_ok=True)

        # 샘플 서브디렉토리
        for subdir in self.samples_subdirs.values():
            os.makedirs(subdir, exist_ok=True)

    def get_sample_path(self, epoch, batch_idx, sample_type):
        """샘플 이미지 저장 경로 생성"""
        return os.path.join(
            self.samples_dir,
            "training",
            f"epoch_{epoch:03d}_batch_{batch_idx:05d}_{sample_type}.jpg"
        )


class VirtualKITTI2Dataset(Dataset):
    def __init__(self, data_dir, camera_id=0, width=640, height=480):
        self.data_dir = Path(data_dir)
        self.camera_id = camera_id
        self.width = width
        self.height = height

        self.rgb_transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.depth_transform = T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR)

        self.scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
        self.scenarios = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right',
                          'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']

        self.rgb_paths, self.depth_paths, self.scene_info = [], [], []
        for scene in self.scenes:
            for scenario in self.scenarios:
                scene_path = self.data_dir / scene / scenario
                rgb_path = scene_path / f"frames/rgb/Camera_{camera_id}"
                depth_path = scene_path / f"frames/depth/Camera_{camera_id}"

                if rgb_path.exists() and depth_path.exists():
                    rgb_files = sorted(rgb_path.glob("*.jpg"))
                    depth_files = sorted(depth_path.glob("*.png"))

                    self.rgb_paths.extend(rgb_files)
                    self.depth_paths.extend(depth_files)
                    self.scene_info.extend([(scene, scenario)] * len(rgb_files))

        assert len(self.rgb_paths) == len(self.depth_paths), "RGB/Depth 파일 수 불일치"

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        try:
            rgb_img = Image.open(self.rgb_paths[idx]).convert('RGB')
            rgb_tensor = self.rgb_transform(rgb_img)

            depth_img = Image.open(self.depth_paths[idx])
            depth_np = np.array(depth_img).astype(np.float32) / 256.0  # VKITTI는 16bit PNG → m 단위
            depth_resized = self.depth_transform(Image.fromarray(depth_np))
            depth_tensor = torch.from_numpy(np.array(depth_resized)).float().unsqueeze(0)

            return {
                'rgb': rgb_tensor,
                'depth': depth_tensor,
                'rgb_path': str(self.rgb_paths[idx]),
                'depth_path': str(self.depth_paths[idx]),
                'scene': self.scene_info[idx][0],
                'scenario': self.scene_info[idx][1],
                'tag': 'vkitti'
            }
        except Exception as e:
            print(f"[VKITTI ERROR] idx={idx} | {e}")
            return self.__getitem__((idx + 1) % len(self))


class HyperSimDataset(Dataset):
    def __init__(self, data_dir, metadata_csv_path, width=640, height=480, rank=0, cache_file="hypersim_cache.json"):
        self.data_dir = Path(data_dir)
        self.metadata_csv_path = metadata_csv_path
        self.width = width
        self.height = height
        self.rank = rank
        self.cache_file = Path(cache_file)

        self.rgb_transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.depth_transform = T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR)

        self.rgb_paths = []
        self.depth_paths = []
        self.scene_info = []

        if self.cache_file.exists():
            self._load_from_cache()
        else:
            self._build_dataset()
            if self.rank == 0:
                self._save_to_cache()

    def _load_from_cache(self):
        import json
        with open(self.cache_file, 'r') as f:
            cache = json.load(f)
            self.rgb_paths = cache['rgb_paths']
            self.depth_paths = cache['depth_paths']
            self.scene_info = cache['scene_info']

    def _save_to_cache(self):
        import json
        cache = {
            'rgb_paths': self.rgb_paths,
            'depth_paths': self.depth_paths,
            'scene_info': self.scene_info
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f)

    def _build_dataset(self):
        import pandas as pd
        df = pd.read_csv(self.metadata_csv_path)
        df = df[(df['included_in_public_release'] == True) & (df['split_partition_name'] == 'train')]
        grouped = df.groupby(['scene_name', 'camera_name'])

        for (scene_name, camera_name), group in grouped:
            rgb_dir = self.data_dir / scene_name / "images" / f"scene_{camera_name}_final_preview"
            depth_dir = self.data_dir / scene_name / "images" / f"scene_{camera_name}_geometry_preview"

            if not rgb_dir.exists() or not depth_dir.exists():
                continue

            for _, row in group.iterrows():
                frame_id = int(row['frame_id'])
                rgb_name = f"frame.{frame_id:04d}.color.jpg"
                depth_name = f"frame.{frame_id:04d}.depth_meters.png"

                rgb_path = rgb_dir / rgb_name
                depth_path = depth_dir / depth_name

                if rgb_path.exists() and depth_path.exists():
                    self.rgb_paths.append(str(rgb_path))
                    self.depth_paths.append(str(depth_path))
                    self.scene_info.append((scene_name, camera_name))

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        try:
            rgb_img = Image.open(self.rgb_paths[idx]).convert('RGB')
            rgb_tensor = self.rgb_transform(rgb_img)

            depth_img = Image.open(self.depth_paths[idx]).convert('L')
            depth_np = np.array(depth_img).astype(np.float32)
            depth_resized = self.depth_transform(Image.fromarray(depth_np))
            depth_tensor = torch.from_numpy(np.array(depth_resized)).float().unsqueeze(0)

            return {
                'rgb': rgb_tensor,
                'depth': depth_tensor,
                'rgb_path': self.rgb_paths[idx],
                'depth_path': self.depth_paths[idx],
                'scene': self.scene_info[idx][0],
                'scenario': self.scene_info[idx][1],
                'tag': 'hypersim'
            }
        except Exception as e:
            print(f"[HyperSim ERROR] idx={idx} | {e}")
            return self.__getitem__((idx + 1) % len(self))


# class SynscapesExrDataset(Dataset):
#     """
#     Synscapes: RGB(PNG/JPG) + Depth(EXR) 전용 Dataset
#       root/
#         ├─ img/    frame_000001.png …
#         └─ depth/  frame_000001.exr …
#     * 깊이: EXR float32(m) 1‑채널 (‘Z' 또는 'R')
#     * 출력: dict(rgb, depth, tag, rgb_path, depth_path)
#             - rgb:   Tensor (3,H,W)  float32, 0‑1, ImageNet norm
#             - depth: Tensor (1,H,W)  float32, [m]
#     """
#     def __init__(self, root_dir, width=640, height=480):
#         self.root = Path(root_dir)
#         self.rgb_paths   = sorted((self.root / "img").glob("*.[jp][pn]g"))
#         self.depth_paths = sorted((self.root / "depth").glob("*.exr"))
#         assert len(self.rgb_paths) == len(self.depth_paths), \
#             f"RGB({len(self.rgb_paths)})·Depth({len(self.depth_paths)}) 수 불일치"
#
#         # 변환기
#         self.resize_rgb   = T.Resize((height, width))
#         self.resize_depth = T.Resize((height, width),
#                                      interpolation=T.InterpolationMode.BILINEAR)
#         self.to_tensor  = T.ToTensor()
#         self.normalize  = T.Normalize([0.485, 0.456, 0.406],
#                                       [0.229, 0.224, 0.225])
#
#     # ---------- EXR 로더 ---------------------------------------------------
#     @staticmethod
#     def _read_exr(path: Path) -> np.ndarray:
#         """
#         OpenEXR single‑channel(Z/R) → np.ndarray(H,W) float32 [meter]
#         """
#         exr = OpenEXR.InputFile(str(path))
#         header = exr.header()
#         W = header['dataWindow'].max.x + 1
#         H = header['dataWindow'].max.y + 1
#
#         # Synscapes EXR 는 'R' 채널 하나만 포함
#         channel_name = 'R' if 'R' in header['channels'] else 'Z'
#         pt_float = Imath.PixelType(Imath.PixelType.FLOAT)
#         depth_str = exr.channel(channel_name, pt_float)
#         depth = np.frombuffer(depth_str, dtype=np.float32).reshape(H, W)
#         return depth  # (H,W) float32 [m]
#
#     # ----------------------------------------------------------------------
#     def __len__(self):
#         return len(self.rgb_paths)
#
#     def __getitem__(self, idx):
#         # --- RGB ---
#         rgb_img = Image.open(self.rgb_paths[idx]).convert("RGB")
#         rgb = self.to_tensor(self.resize_rgb(rgb_img))
#         rgb = self.normalize(rgb)
#
#         # --- Depth(EXR) ---
#         depth_np = self._read_exr(self.depth_paths[idx])          # (H,W)
#         depth_pil = Image.fromarray(depth_np)                     # tmp PIL
#         depth = self.resize_depth(depth_pil)[None]                # (1,H,W)
#
#         return {
#             "rgb":       rgb,
#             "depth":     depth,            # float32, m 단위
#             "tag":       "synscapes",
#             "rgb_path":  str(self.rgb_paths[idx]),
#             "depth_path": str(self.depth_paths[idx]),
#         }



class TaggedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tag):
        self.dataset = dataset
        self.tag = tag

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample['tag'] = self.tag
        return sample


class RGBDTransform:
    def __init__(self, output_size=(320, 320), crop_size=(288, 288), crop_prob=0.8, flip_prob=0.5,
                 color_jitter_params=None, color_jitter_prob=0.5, add_noise=False, noise_prob=0.5):
        self.output_size = output_size
        self.crop_size = crop_size
        self.crop_prob = crop_prob
        self.flip_prob = flip_prob
        self.color_jitter = None
        if color_jitter_params:
            self.color_jitter = T.RandomApply(
                [T.ColorJitter(**color_jitter_params)],
                p=color_jitter_prob
            )
        self.add_noise = add_noise
        self.noise_prob = noise_prob

    def __call__(self, rgb, depth):
        # Random horizontal flip
        if random.random() < self.flip_prob:
            rgb = torch.flip(rgb, dims=[2])  # width 방향
            depth = torch.flip(depth, dims=[2])

        # Random crop
        if random.random() < self.crop_prob:
            B, C, H, W = rgb.shape  # 수정: batch 포함해서 shape 받기
            crop_h, crop_w = self.crop_size
            if H > crop_h and W > crop_w:
                top = random.randint(0, H - crop_h)
                left = random.randint(0, W - crop_w)
                rgb = rgb[:, :, top:top + crop_h, left:left + crop_w]
                depth = depth[:, :, top:top + crop_h, left:left + crop_w]

        # Resize (훈련용 320×320 통일)
        rgb = T.functional.resize(rgb, self.output_size, antialias=True)
        depth = T.functional.resize(depth, self.output_size, antialias=True)

        # Color jitter (RGB만)
        if self.color_jitter:
            rgb = self.color_jitter(rgb)

        # Gaussian noise (RGB만)
        if self.add_noise and random.random() < self.noise_prob:
            noise = torch.randn_like(rgb) * 0.02
            rgb = rgb + noise
            rgb = torch.clamp(rgb, 0.0, 1.0)

        return rgb, depth


# class RatioDistributedSampler(DistributedSampler):
#     def __init__(self, dataset, vkitti_size, hypersim_size,
#                  num_replicas=None, rank=None, shuffle=True, seed=0):
#         super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
#
#         self.vkitti_size = vkitti_size
#         self.hypersim_size = hypersim_size
#         self.total_size = vkitti_size + hypersim_size
#
#         self.hypersim_per_rank = hypersim_size // self.num_replicas
#         self.vkitti_per_rank = int(self.hypersim_per_rank * 1 / 9)
#
#         self.num_samples = self.vkitti_per_rank + self.hypersim_per_rank
#
#         print(f"\n[Rank {rank}] RatioSampler:")
#         print(f"  - vkitti per rank    = {self.vkitti_per_rank}")
#         print(f"  - hypersim per rank  = {self.hypersim_per_rank}")
#         print(f"  - total per rank     = {self.num_samples}")
#
#     def __iter__(self):
#         g = torch.Generator()
#         g.manual_seed(self.seed + self.epoch)
#
#         # 랜덤 인덱스 생성
#         vkitti_indices = torch.randperm(self.vkitti_size, generator=g).tolist()
#         hypersim_indices = torch.randperm(self.hypersim_size, generator=g).tolist()
#
#         start_h = self.rank * self.hypersim_per_rank
#         end_h = start_h + self.hypersim_per_rank
#         start_v = self.rank * self.vkitti_per_rank
#         end_v = start_v + self.vkitti_per_rank
#
#         hs = [i + self.vkitti_size for i in hypersim_indices[start_h:end_h]]
#         vk = vkitti_indices[start_v:end_v]
#
#         mixed = vk + hs
#         random.shuffle(mixed)
#         return iter(mixed)
#
#     def __len__(self):
#         return self.num_samples


class RatioDistributedSampler(DistributedSampler):
    """
    Synscapes + Hypersim 을 1:1 비율로 뽑아주는 Sampler
    (batch_size == 1 환경, DDP 지원)
    """
    def __init__(self, dataset, syn_size, hypersim_size,
                 num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset,
                         num_replicas=num_replicas,
                         rank=rank,
                         shuffle=shuffle,
                         seed=seed)

        self.syn_size      = syn_size
        self.hypersim_size = hypersim_size
        self.total_size    = syn_size + hypersim_size

        # ―――― 1 : 1 비율 ――――
        # 두 도메인 중 작은 쪽에 맞춰 rank‑당 샘플 수를 결정
        base = min(self.syn_size, self.hypersim_size) // self.num_replicas
        self.syn_per_rank      = base
        self.hypersim_per_rank = base

        self.num_samples = self.syn_per_rank + self.hypersim_per_rank

        print(f"\n[Rank {rank}] RatioSampler  (Syn : HS = 1 : 1)")
        print(f"  - synscapes per rank = {self.syn_per_rank}")
        print(f"  - hypersim  per rank = {self.hypersim_per_rank}")
        print(f"  - total     per rank = {self.num_samples}")

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 1) 무작위 인덱스
        syn_indices      = torch.randperm(self.syn_size, generator=g).tolist()
        hypersim_indices = torch.randperm(self.hypersim_size, generator=g).tolist()

        # 2) rank 별 슬라이스
        s0 = self.rank * self.syn_per_rank
        s1 = s0 + self.syn_per_rank
        h0 = self.rank * self.hypersim_per_rank
        h1 = h0 + self.hypersim_per_rank

        syn = syn_indices[s0:s1]
        hs  = [i + self.syn_size               # ← ConcatDataset offset
               for i in hypersim_indices[h0:h1]]

        mixed = syn + hs
        random.shuffle(mixed)
        return iter(mixed)

    def __len__(self):
        return self.num_samples


# def find_latest_checkpoint(checkpoint_dir):
#     """
#     checkpoint_dir 안에서 가장 마지막 epoch 파일을 찾아서 반환
#     """
#     checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
#     if not checkpoint_files:
#         return None
#
#     # "epoch_{숫자}.pth" 형태에서 숫자만 추출
#     epochs = []
#     for file in checkpoint_files:
#         match = re.search(r"epoch_(\d+).pth", file)
#         if match:
#             epochs.append((int(match.group(1)), file))
#
#     if not epochs:
#         return None
#
#     # epoch 번호 기준으로 최대값 찾기
#     latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
#     return latest_file
import re

def find_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not ckpt_files:
        return None

    # latest: epoch_10.pth
    latest_pattern = re.compile(r"^epoch_(\d+)\.pth$")
    # best: best_epoch_18_0.003824.pth
    best_pattern = re.compile(r"^best_epoch_(\d+)_([0-9.]+)\.pth$")

    latest_matches = []
    best_matches = []

    for f in ckpt_files:
        m1 = latest_pattern.match(f)
        if m1:
            epoch = int(m1.group(1))
            latest_matches.append((epoch, f))
            continue

        m2 = best_pattern.match(f)
        if m2:
            epoch = int(m2.group(1))
            loss = float(m2.group(2))
            best_matches.append((loss, f))

    if latest_matches:
        # epoch 숫자가 가장 큰 latest 선택
        _, best_file = max(latest_matches, key=lambda x: x[0])
        return os.path.join(ckpt_dir, best_file)

    if best_matches:
        # loss가 가장 낮은 best 선택
        _, best_file = min(best_matches, key=lambda x: x[0])
        return os.path.join(ckpt_dir, best_file)

    return None




def modify_unet_input_channels(unet, original_channels=4, target_channels=8):
    conv_in = unet.conv_in
    original_weight = conv_in.weight.data

    new_conv = torch.nn.Conv2d(
        target_channels,
        conv_in.out_channels,
        kernel_size=conv_in.kernel_size,
        stride=conv_in.stride,
        padding=conv_in.padding,
    )

    with torch.no_grad():
        new_weight = torch.zeros(
            conv_in.out_channels,
            target_channels,
            *conv_in.kernel_size
        )
        new_weight[:, :original_channels] = original_weight
        new_weight[:, original_channels:] = original_weight  # 복사해서 채우기
        new_conv.weight.data = new_weight
        new_conv.bias.data = conv_in.bias.data

    unet.conv_in = new_conv
    return unet


def load_pipeline(rank):
    # Stable Diffusion v1.5 or v2
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=torch.float16,  # FP16으로 변경
        use_safetensors=True,
    ).to(rank)

    vae = pipe.vae
    unet = pipe.unet

    # VAE를 FP16으로 변환
    vae = vae.half()
    
    # 🔧 unet 입력 채널 확장
    unet = expand_unet_input_channels(unet, new_in_channels=8)
    
    # UNet을 FP16으로 변환
    unet = unet.half()

    return vae, unet


def prepare_dataloader(rank, world_size, width, height):
    vkitti_dir = "/media/hspark/My Passport/Virtual Kitti 2"
    hypersim_dir = "/media/hspark/My Passport2/downloads"
    metadata_csv_path = "./metadata_images_split_scene_v1.csv"

    vkitti_dataset = VirtualKITTI2Dataset(vkitti_dir, camera_id=0, width=width, height=height)
    hypersim_dataset = HyperSimDataset(
        data_dir=hypersim_dir,
        metadata_csv_path=metadata_csv_path,
        width=width,
        height=height,
        rank=rank
    )

    # Tagging
    vkitti_tagged = TaggedDataset(vkitti_dataset, "vkitti")
    hypersim_tagged = TaggedDataset(hypersim_dataset, "hypersim")
    combined_dataset = torch.utils.data.ConcatDataset([vkitti_tagged, hypersim_tagged])

    sampler = RatioDistributedSampler(
        dataset=combined_dataset,
        vkitti_size=len(vkitti_dataset),
        hypersim_size=len(hypersim_dataset),
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    return dataloader


def normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    """
    Normalize a depth map tensor to the range [-1, 1] using valid (non-zero) values.

    Args:
        depth (torch.Tensor): [B, 1, H, W] tensor

    Returns:
        torch.Tensor: normalized depth in [-1, 1]
    """
    B, _, H, W = depth.shape
    depth_flat = depth.view(B, -1)

    # 유효한 값만 마스킹
    valid_mask = depth_flat > 0

    # 배치별 d_min, d_max 계산 (유효값 기준)
    d_min = torch.full((B, 1), float('inf'), device=depth.device)
    d_max = torch.full((B, 1), float('-inf'), device=depth.device)

    for b in range(B):
        valid = depth_flat[b][valid_mask[b]]
        if valid.numel() > 0:
            d_min[b] = valid.min()
            d_max[b] = valid.max()
        else:
            d_min[b] = 0.0
            d_max[b] = 1.0

    # [B, 1, 1, 1] 형태로 broadcasting
    d_min = d_min.view(B, 1, 1, 1)
    d_max = d_max.view(B, 1, 1, 1)

    # 정규화: [-1, 1]로 스케일링
    normalized = (depth - d_min) / (d_max - d_min + 1e-6)
    normalized = normalized * 2 - 1
    return normalized.clamp(-1, 1)


def prepare_dataloader_hypersim(rank, world_size, width, height):
    hypersim_dir = "/media/hspark/My Passport2/downloads"
    metadata_csv_path = "./metadata_images_split_scene_v1.csv"

    hypersim_dataset = HyperSimDataset(
        data_dir=hypersim_dir,
        metadata_csv_path=metadata_csv_path,
        width=width,
        height=height,
        rank=rank
    )

    # 태그 부여(시각화·debug용)
    hypersim_tagged = TaggedDataset(hypersim_dataset, "hypersim")

    # 분산 학습용 샘플러 – VKITTI가 없으므로 기본 DistributedSampler로 충분
    sampler = DistributedSampler(
        hypersim_tagged,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = torch.utils.data.DataLoader(
        hypersim_tagged,
        batch_size=1,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    return dataloader


# # ------------------------------------------------------------------------
# #  prepare_dataloader_synscapes_hypersim
# # ------------------------------------------------------------------------
# def prepare_dataloader_synscapes_hypersim(
#         rank: int,
#         world_size: int,
#         syn_root: str = "/media/hspark/My Passport2/synscapes",
#         hs_root: str  = "/media/hspark/My Passport2/downloads",
#         hs_meta_csv: str = "./metadata_images_split_scene_v1.csv",
#         width: int = 640,
#         height: int = 480,
#         num_workers: int = 4,
#         batch_size: int = 1,
# ) -> torch.utils.data.DataLoader:
#     """
#     Synscapes(EXR) + Hypersim DataLoader (1:1, batch_size=1)
#     * DDP rank/world_size를 인자로 받아 분산 학습 대응
#     * RatioDistributedSampler로 두 도메인을 균형 있게 섞음
#     """
#
#     # 1) Dataset -----------------------------------------------------------
#     syn_dataset = SynscapesExrDataset(syn_root, width, height)
#     hs_dataset  = HyperSimDataset(hs_root, hs_meta_csv, width, height, rank=rank)
#
#     # 2) 태그 부여 (시각화·디버깅용) ----------------------------------------
#     syn_tagged = TaggedDataset(syn_dataset, "synscapes")
#     hs_tagged  = TaggedDataset(hs_dataset,  "hypersim")
#
#     # 3) ConcatDataset  (Synscapes를 반드시 앞에!) -------------------------
#     combined_dataset = torch.utils.data.ConcatDataset([syn_tagged, hs_tagged])
#
#     # 4) RatioDistributedSampler (1 : 1) -----------------------------------
#     sampler = RatioDistributedSampler(
#         dataset=combined_dataset,
#         syn_size=len(syn_dataset),
#         hypersim_size=len(hs_dataset),
#         num_replicas=world_size,
#         rank=rank,
#         shuffle=True,
#     )
#
#     # 5) DataLoader --------------------------------------------------------
#     loader = torch.utils.data.DataLoader(
#         combined_dataset,
#         batch_size=batch_size,      # grad‑accum 으로 실질 배치↑
#         sampler=sampler,
#         num_workers=num_workers,
#         pin_memory=True,
#         persistent_workers=(num_workers > 0),
#     )
#
#     if rank == 0:
#         print(f"[DataLoader] Synscapes {len(syn_dataset)}, "
#               f"Hypersim {len(hs_dataset)}  → per‑epoch {len(loader)} steps")
#
#     return loader



def visualize_batch(batch, output_dir, tag="debug", max_samples=8):
    os.makedirs(output_dir, exist_ok=True)

    rgb_batch = batch['rgb'][:max_samples]
    depth_batch = batch['depth'][:max_samples]
    tag_batch = batch['tag'][:max_samples]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(len(rgb_batch)):
        tag_name = tag_batch[i]
        rgb = rgb_batch[i].cpu() * std + mean
        rgb = rgb.clamp(0, 1)
        rgb_pil = TF.to_pil_image(rgb)
        rgb_pil.save(os.path.join(output_dir, f"{tag}_rgb_{i}.png"))

        depth = depth_batch[i, 0].cpu().numpy()
        valid_mask = depth > 0
        valid_depth = depth[valid_mask]

        if len(valid_depth) > 0:
            d_min, d_max = valid_depth.min(), valid_depth.max()
            unique_vals = np.unique(valid_depth)
            n_unique = len(unique_vals)
            step = np.median(np.diff(np.sort(unique_vals))) if n_unique > 1 else 0

            is_integer_quantized = (
                d_max <= 256 and
                np.all(np.abs(valid_depth - np.round(valid_depth)) < 1e-3) and
                step >= 1
            )
            is_skewed = d_max > 1000 or (np.percentile(valid_depth, 90) / np.percentile(valid_depth, 10 + 1e-6)) > 10
        else:
            d_min, d_max = 0, 0
            is_integer_quantized = False
            is_skewed = False

        # ✅ vmin/vmax 자동 설정
        if is_integer_quantized:
            vmin, vmax = np.min(valid_depth), np.max(valid_depth)
        else:
            vmin, vmax = np.percentile(valid_depth, 2), np.percentile(valid_depth, 98)

        # ✅ 시각화용 변환
        depth_clipped = np.clip(depth, vmin, vmax)

        # ▶️ log 스케일링 (VKITTI만)
        if tag_name == "vkitti":
            eps = 1e-3
            depth_scaled = np.log(depth_clipped + eps)
            depth_scaled = (depth_scaled - np.min(depth_scaled)) / (np.max(depth_scaled) - np.min(depth_scaled) + 1e-6)
            depth_norm = 1.0 - depth_scaled
        else:
            depth_norm = (depth_clipped - vmin) / (vmax - vmin + 1e-6)
            depth_norm = 1.0 - depth_norm

        # 🔥 시각화
        depth_colored = plt.get_cmap("magma")(depth_norm)[..., :3]
        depth_colored = (depth_colored * 255).astype(np.uint8)
        plt.imsave(os.path.join(output_dir, f"{tag}_depth_{i}_{tag_name}.png"), depth_colored)

        valid_ratio = 100.0 * np.sum(valid_mask) / valid_mask.size
        print(f"[Debug] {tag}_{i} | Tag: {tag_name}")
        print(f" - Depth range: {d_min:.3f} ~ {d_max:.3f}")
        print(f" - Valid pixel ratio: {valid_ratio:.2f}%")
        print(f" - Unique values: {n_unique}, Median step: {step:.3f}")
        print(f" - vmin/vmax used: {vmin:.2f} / {vmax:.2f}")
        print(f" 🔎 Inference: {'정수 양자화 (Integer Quantized)' if is_integer_quantized else ''}"
              f"{'정규화 문제 (Skewed)' if is_skewed else ''}"
              f"{'정상 (Normal)' if not is_integer_quantized and not is_skewed else ''}")


# class ConsistencyModel(nn.Module):
#     def __init__(self, unet, alphas_cumprod):
#         super().__init__()
#         self.unet = unet
#         self.alphas_cumprod = alphas_cumprod  # torch.Tensor shape [num_steps]
#
#         # (추가) Unet 출력이 4채널이 아닐 경우 대비해서 1x1 conv projection
#         self.out_proj = nn.Conv2d(4, 4, kernel_size=1)  # 입력 4채널, 출력 4채널 (사실상 패스, 대비용)
#
#     def forward(self, z, t, w=1.0):
#         """
#         z: (B, 8, H, W) -> [image 4ch + noisy depth 4ch]
#         t: (B,)
#         w: guidance scale (기본 1.0)
#         """
#         dummy_text_embeds = torch.zeros(
#             (z.shape[0], 77, 1024),  # (batch_size, seq_len, hidden_dim)
#             device=z.device,
#             dtype=z.dtype
#         )
#
#         # Unet 통과
#         noise_pred = self.unet(z, t, encoder_hidden_states=dummy_text_embeds).sample  # (B, 4, H, W)
#         # (여기서는 output이 4채널로 나오는 걸 기대함)
#
#         # depth latent만 복원하는 걸로 바로 사용
#         pred_depth_latent = noise_pred  # (B, 4, H, W)
#
#         return pred_depth_latent

# ConsistencyModel 수정 (gradient checkpoint 버전)
class ConsistencyModel(nn.Module):
    def __init__(self, unet, alphas_cumprod, use_gradient_checkpointing=False):
        super().__init__()
        self.unet = unet
        self.alphas_cumprod = alphas_cumprod
        self.use_gradient_checkpointing = use_gradient_checkpointing  # ✨ 추가

    def forward(self, z, t, w=1.0):
        dummy_text_embeds = torch.zeros(
            (z.shape[0], 77, 1024),
            device=z.device,
            dtype=z.dtype
        )

        if self.use_gradient_checkpointing:
            # ✨ gradient checkpoint 적용
            def custom_forward(*inputs):
                z, t, encoder_hidden_states = inputs
                return self.unet(z, t, encoder_hidden_states=encoder_hidden_states).sample

            noise_pred = checkpoint.checkpoint(custom_forward, z, t, dummy_text_embeds, use_reentrant=False)
        else:
            # ✨ 기본 forward
            noise_pred = self.unet(z, t, encoder_hidden_states=dummy_text_embeds).sample

        return noise_pred


def get_alphas_cumprod(beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=1000):
    betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod


def get_sqrt_alpha_t(t, alphas_cumprod):
    """
    t: (B,) long tensor
    alphas_cumprod: (T,) tensor
    """
    return alphas_cumprod[t].sqrt().unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B,1,1,1)

def get_sqrt_one_minus_alpha_t(t, alphas_cumprod):
    """
    1 - alpha_cumprod에서 sqrt
    """
    return (1. - alphas_cumprod[t]).sqrt().unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B,1,1,1)


# @torch.no_grad()
# def sample_and_save_depth_model(model, vae, batch, device, batch_idx, output_dir, alphas_cumprod, epoch, num_inference_steps=4):
#     model.eval()
#     vae.eval()
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     rgb = batch["rgb"].to(device)          # (B, 3, H, W)
#     depth = batch["depth"].to(device)       # (B, 1, H, W)
#     tag_batch = batch["tag"]
#
#     # VAE encode
#     image_latent = vae.encode(rgb).latent_dist.sample() * 0.18215
#     depth_expanded = depth.repeat(1, 3, 1, 1)
#     depth_latent = vae.encode(depth_expanded).latent_dist.sample() * 0.18215
#
#     # Set initial noisy latent
#     B = rgb.size(0)
#     T = alphas_cumprod.shape[0]   # e.g., 1000
#     # print(T) # 1000
#     t = torch.full((B,), T-1, device=device, dtype=torch.long)
#
#     noise = torch.randn_like(depth_latent)
#     noisy_depth_latent = add_noise_to_depth_latent(depth_latent, noise, t, alphas_cumprod)
#     noisy_latent = torch.cat([image_latent, noisy_depth_latent], dim=1)
#
#     # 4-step denoising (fixed ratio)
#     for ratio in [1.0, 0.66, 0.33, 0.0]:
#         t_scaled = (t.float() * ratio).round().long()
#         z0_pred = model(noisy_latent, t_scaled)
#         noisy_latent[:, 4:] = z0_pred
#
#     # Final prediction
#     pred_depth_latent = noisy_latent[:, 4:]
#     pred_depth = vae.decode(pred_depth_latent / 0.18215).sample
#     pred_depth_first_channel = pred_depth[:, 0:1, :, :]
#
#     # Prepare normalization constants
#     mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device).view(3, 1, 1)
#
#     for i in range(B):
#         tag_name = tag_batch[i]
#
#         # Restore RGB
#         rgb_img = rgb[i]
#         rgb_img = rgb_img * std + mean
#         rgb_img = rgb_img.clamp(0, 1)
#         rgb_np = (rgb_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#
#         # Restore Depth
#         pred_depth_np = pred_depth_first_channel[i, 0].cpu().numpy()
#         valid_mask = pred_depth_np > 0
#         valid_depth = pred_depth_np[valid_mask]
#
#         if len(valid_depth) > 0:
#             vmin, vmax = np.percentile(valid_depth, 2), np.percentile(valid_depth, 98)
#         else:
#             vmin, vmax = 0.0, 1.0
#
#         depth_clipped = np.clip(pred_depth_np, vmin, vmax)
#
#         if tag_name == "vkitti":
#             eps = 1e-3
#             depth_scaled = np.log(depth_clipped + eps)
#             depth_scaled = (depth_scaled - depth_scaled.min()) / (depth_scaled.max() - depth_scaled.min() + 1e-6)
#         else:
#             depth_scaled = (depth_clipped - vmin) / (vmax - vmin + 1e-6)
#
#         depth_norm = 1.0 - depth_scaled  # 먼 거리 어둡게
#         depth_colored = plt.get_cmap("magma")(depth_norm)[..., :3]
#         depth_colored = (depth_colored * 255).astype(np.uint8)
#
#         # Combine RGB and Depth
#         combined = np.concatenate([rgb_np, depth_colored], axis=1)
#
#         # Save
#         save_path = os.path.join(output_dir, f"sample_{epoch}_{batch_idx:05d}_{i}_{tag_name}.png")
#         plt.imsave(save_path, combined)


@torch.no_grad()
def sample_and_save_depth_model(model, vae, batch, device, batch_idx, output_dir, alphas_cumprod, epoch, num_inference_steps=4):
    """
    LCM 모델에서 pred_z0과 depth_latent를 이미지로 저장하는 버전 통합
    """
    model.eval()
    vae.eval()

    os.makedirs(output_dir, exist_ok=True)

    rgb = batch["rgb"].to(device)  # (B, 3, H, W)
    depth = batch["depth"].to(device)  # (B, 1, H, W)
    tag_batch = batch["tag"]

    B = rgb.shape[0]

    # ✨ VAE encode (autocast)
    with torch.cuda.amp.autocast():
        image_latent = vae.encode(rgb).latent_dist.sample() * 0.18215
        depth_expanded = depth.repeat(1, 3, 1, 1)
        depth_latent = vae.encode(depth_expanded).latent_dist.sample() * 0.18215

    T = alphas_cumprod.shape[0]
    t = torch.full((B,), T - 1, device=device, dtype=torch.long)

    sqrt_alpha = get_sqrt_alpha_t(t, alphas_cumprod)
    sqrt_one_minus_alpha = get_sqrt_one_minus_alpha_t(t, alphas_cumprod)

    noise = torch.randn_like(depth_latent)
    noisy_depth_latent = sqrt_alpha * depth_latent + sqrt_one_minus_alpha * noise
    noisy_latent = torch.cat([image_latent, noisy_depth_latent], dim=1)

    # ✨ 4-step inference
    for ratio in [1.0, 0.66, 0.33, 0.0]:
        t_scaled = (t.float() * ratio).round().long()
        pred_z0 = model(noisy_latent, t_scaled)
        noisy_latent[:, 4:] = pred_z0

    pred_depth_latent = noisy_latent[:, 4:]  # (B, 4, H, W)

    # --- pred_z0 vs depth_latent 채널별 비교 저장 ---
    save_latent_comparison(
        pred_z0=pred_depth_latent[0:1],  # 첫 번째 샘플만
        depth_latent=depth_latent[0:1],
        save_path=f"{output_dir}/latent_comp_epoch{epoch}_batch{batch_idx}.png"
    )

    # --- 복원된 Depth 디코딩 ---
    pred_depth = vae.decode(pred_depth_latent / 0.18215).sample
    pred_depth_first_channel = pred_depth[:, 0:1, :, :]

    # ---복원된 GT Depth 디코딩 ---
    gt_depth_img = vae.decode(depth_latent / 0.18215).sample  # B×3×H×W
    gt_depth_first_channel = gt_depth_img[:, 0:1, :, :]

    # ✨ Prepare normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    for i in range(B):
        tag_name = tag_batch[i]

        # === 복원된 RGB ===
        rgb_img = rgb[i]
        rgb_img = rgb_img * std + mean  # de-normalize
        rgb_img = rgb_img.clamp(0, 1)
        rgb_np = (rgb_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # === 복원된 Depth ===
        pred_depth_np = pred_depth_first_channel[i, 0].cpu().numpy()
        valid_mask = pred_depth_np > 0
        valid_depth = pred_depth_np[valid_mask]

        # === 복원된 GT Depth ===
        gt_depth_np = gt_depth_first_channel[i, 0].cpu().numpy()
        # valid_mask_gt = gt_depth_np > 0
        # valid_depth_gt = gt_depth_np[valid_mask_gt]

        if len(valid_depth) > 0:
            vmin, vmax = np.percentile(valid_depth, 2), np.percentile(valid_depth, 98)
        else:
            vmin, vmax = 0.0, 1.0

        depth_clipped = np.clip(pred_depth_np, vmin, vmax)
        gt_depth_clipped = np.clip(gt_depth_np, vmin, vmax)

        if tag_name == "vkitti":
            eps = 1e-3
            depth_scaled = np.log(depth_clipped + eps)
            depth_scaled = (depth_scaled - depth_scaled.min()) / (depth_scaled.max() - depth_scaled.min() + 1e-6)
        else:
            depth_scaled = (depth_clipped - vmin) / (vmax - vmin + 1e-6)
            gt_depth_scaled = (gt_depth_clipped - vmin) / (vmax - vmin + 1e-6)

        depth_norm = 1.0 - depth_scaled  # 먼 거리 어둡게
        depth_colored = plt.get_cmap("turbo")(depth_norm)[..., :3]  # (H, W, 3) #magma, rainbow, jet, viridis, turbo
        depth_colored = (depth_colored * 255).astype(np.uint8)
        gt_depth_norm = 1.0 - gt_depth_scaled  # 먼 거리 어둡게
        gt_depth_colored = plt.get_cmap("turbo")(gt_depth_norm)[..., :3]  # (H, W, 3) #magma, rainbow, jet, viridis, turbo
        gt_depth_colored = (gt_depth_colored * 255).astype(np.uint8)

        # === RGB + Depth 가로로 이어붙이기 ===
        combined = np.concatenate([rgb_np, depth_colored, gt_depth_colored], axis=1)  # (H, W*2, 3)

        # === Save ===
        save_path = os.path.join(output_dir, f"sample_epoch{epoch}_batch{batch_idx}_idx{i}_{tag_name}.png")
        plt.imsave(save_path, combined)


def visualize_depth(depth_map, clamp_min=None, clamp_max=None):
    """
    depth_map: (1, H, W) 텐서 (1채널)
    output: (3, H, W) 텐서 (normalize된 3채널)
    """
    # depth_map: (1, H, W) → (H, W)
    depth_map = depth_map.squeeze(0)

    # 자동 min-max 클리핑
    if clamp_min is None:
        clamp_min = depth_map.min()
    if clamp_max is None:
        clamp_max = depth_map.max()

    depth_map = depth_map.clamp(min=clamp_min, max=clamp_max)

    # 0 ~ 1 normalize
    depth_map = (depth_map - clamp_min) / (clamp_max - clamp_min + 1e-8)

    # 3채널로 복제 (grayscale)
    depth_map_3ch = depth_map.unsqueeze(0).repeat(3, 1, 1)

    return depth_map_3ch


# 🔥 추가 함수: latent 비교용
def save_latent_comparison(pred_z0, depth_latent, save_path):
    """
    pred_z0, depth_latent를 채널별로 시각화해서 저장
    """
    pred_z0 = pred_z0.squeeze(0)  # (4, H, W)
    depth_latent = depth_latent.squeeze(0)  # (4, H, W)

    def normalize(x):
        x_min = x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + 1e-8)

    pred_z0_norm = normalize(pred_z0)
    depth_latent_norm = normalize(depth_latent)

    pred_z0_3ch = pred_z0_norm.unsqueeze(1).repeat(1, 3, 1, 1)  # (4, 3, H, W)
    depth_latent_3ch = depth_latent_norm.unsqueeze(1).repeat(1, 3, 1, 1)  # (4, 3, H, W)

    pred_row = torch.cat([pred_z0_3ch[i] for i in range(4)], dim=2)  # (3, H, 4*W)
    depth_row = torch.cat([depth_latent_3ch[i] for i in range(4)], dim=2)  # (3, H, 4*W)

    final_image = torch.cat([pred_row, depth_row], dim=1)  # (3, 2*H, 4*W)

    vutils.save_image(final_image, save_path, normalize=False)


def expand_to_3channel(x):
    """
    (B, 4, H, W) → (B*4, 3, H, W)
    각 채널을 3채널로 복제
    """
    B, C, H, W = x.shape
    assert C == 4, f"Expected 4 channels, got {C}"
    
    # (B, 4, H, W) → (B*4, 1, H, W)
    expanded = x.view(B * C, 1, H, W)
    
    # (B*4, 1, H, W) → (B*4, 3, H, W) - 각 채널을 3번 복제
    expanded = expanded.repeat(1, 3, 1, 1)
    
    return expanded


def extract(a, t, shape):
    """
    a: (T,) - 전체 타임스텝별 alphas_cumprod
    t: (B,) - 배치마다 선택된 타임스텝
    shape: (B, 1, 1, 1) - 원하는 리쉐입 형태
    """
    out = a.gather(-1, t)
    return out.view(shape).float()


def train_lcm_one_epoch(
    model,
    vae,
    optimizer,
    train_loader,
    device,
    alphas_cumprod,
    epoch,
    output_dir,
    accumulation_steps=16,
    save_interval=500,  # 샘플 저장 주기 (batch 기준)
):

    optimizer.zero_grad(set_to_none=True)

    model.train()
    scaler = GradScaler(enabled=True)  # FP16 스케일링 활성화
    total_loss = 0.0
    ema_loss = None
    num_batches = len(train_loader)
    transform = RGBDTransform(
        output_size=(320, 320),
        crop_size=(288, 288),
        crop_prob=0.5,
        flip_prob=0.5,
        color_jitter_params={
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.05,
            "hue": 0.05,
        },
        color_jitter_prob=0.5,
        add_noise=True,
        noise_prob=0.5
    )

    if dist.get_rank() == 0:
        progress_bar = tqdm(total=num_batches, desc=f"[Epoch {epoch}]")

    for batch_idx, batch in enumerate(train_loader):
        rgb = batch["rgb"].to(device, dtype=torch.float16)  # FP16으로 변환
        depth = batch["depth"].to(device, dtype=torch.float16)  # FP16으로 변환
        rgb, depth = transform(rgb, depth)

        B = rgb.shape[0]
        T = alphas_cumprod.shape[0]
        t = torch.randint(0, T, (B,), device=device).long()

        with autocast(enabled=True):  # FP16 autocast 활성화
            image_latent = vae.encode(rgb).latent_dist.sample() * 0.18215
            depth_latent = vae.encode(depth.repeat(1, 3, 1, 1)).latent_dist.sample() * 0.18215
            noise = torch.randn_like(depth_latent, dtype=torch.float16)  # FP16 noise

            # NaN 체크
            if torch.isnan(image_latent).any() or torch.isnan(depth_latent).any():
                print(f"[WARNING] NaN detected in latents at batch {batch_idx}, skipping...")
                continue

            sync_context = (
                model.no_sync() if (batch_idx + 1) % accumulation_steps else nullcontext()
            )
            with sync_context:
                loss = compute_lcm_loss(
                    model=model,
                    image_latent=image_latent,
                    depth_latent=depth_latent,
                    noise=noise,
                    t=t,
                    alphas_cumprod=alphas_cumprod,
                ) / accumulation_steps

                # NaN 체크
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARNING] NaN/Inf loss detected at batch {batch_idx}, skipping...")
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss).backward()

        total_loss += loss.item()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping 추가
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # EMA loss 업데이트
        if ema_loss is None:
            ema_loss = loss.item() * accumulation_steps
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * (loss.item() * accumulation_steps)

        # Progress Bar 업데이트
        if dist.get_rank() == 0:
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.8f}",
                "ema_loss": f"{ema_loss:.8f}"
            })

            if batch_idx % save_interval == 0:
                sample_and_save_depth_model(
                    model=model,
                    vae=vae,
                    batch=batch,
                    device=device,
                    batch_idx=batch_idx,
                    output_dir=output_dir,
                    alphas_cumprod=alphas_cumprod,
                    epoch=epoch,
                    num_inference_steps=4,
                )

    if torch.distributed.get_rank() == 0:
        progress_bar.close()

    avg_loss = (total_loss * accumulation_steps) / len(train_loader)
    return avg_loss


def compute_lcm_loss(model, image_latent, depth_latent, noise, t, alphas_cumprod):
    """
    LCM Consistency Loss만 계산
    """
    B = image_latent.shape[0]
    sqrt_alpha = extract(alphas_cumprod, t, (B, 1, 1, 1)).sqrt()
    sqrt_one_minus_alpha = (1. - extract(alphas_cumprod, t, (B, 1, 1, 1))).sqrt()

    # 안전한 연산을 위한 epsilon 추가
    eps = 1e-8
    
    noisy_depth = sqrt_alpha * depth_latent + sqrt_one_minus_alpha * noise
    noisy_concat = torch.cat([image_latent, noisy_depth], dim=1)

    pred_z0 = model(noisy_concat, t)
    
    # NaN 방지를 위한 안전한 MSE loss
    loss_consistency = F.mse_loss(pred_z0, depth_latent)
    
    # NaN/Inf 체크 및 처리
    if torch.isnan(loss_consistency) or torch.isinf(loss_consistency):
        print(f"[WARNING] NaN/Inf in loss_consistency: {loss_consistency}")
        # 안전한 fallback 값 반환
        return torch.tensor(0.1, device=loss_consistency.device, dtype=loss_consistency.dtype, requires_grad=True)
    
    return loss_consistency


def add_noise_lcm_style(latents, noise, timesteps, alphas_cumprod):
    """
    LCM-style 노이즈 추가 방법
    - image latent는 건드리지 않고,
    - depth latent에만 noise를 추가
    """
    B, C, H, W = latents.shape
    assert C == 8, "latents must have 8 channels (4 image + 4 depth)"

    image_latent, depth_latent = latents[:, :4], latents[:, 4:]

    sqrt_alpha = extract(alphas_cumprod, timesteps, (B, 1, 1, 1)).sqrt()
    sqrt_one_minus_alpha = (1. - extract(alphas_cumprod, timesteps, (B, 1, 1, 1))).sqrt()

    noisy_depth_latent = sqrt_alpha * depth_latent + sqrt_one_minus_alpha * noise
    noisy_latents = torch.cat([image_latent, noisy_depth_latent], dim=1)  # 다시 (B, 8, H, W)
    return noisy_latents


def add_noise_to_depth_latent(depth_latent, noise, timesteps, alphas_cumprod):
    """
    depth_latent: (B, 4, H, W)
    noise: (B, 4, H, W)
    timesteps: (B,)
    alphas_cumprod: (T,)
    """
    B, C, H, W = depth_latent.shape
    assert C == 4, "depth_latent must have 4 channels"
    sqrt_alpha = extract(alphas_cumprod, timesteps, (B, 1, 1, 1)).sqrt()
    sqrt_one_minus_alpha = (1. - extract(alphas_cumprod, timesteps, (B, 1, 1, 1))).sqrt()
    return sqrt_alpha * depth_latent + sqrt_one_minus_alpha * noise


@torch.no_grad()
def eval_on_kitti_test_list(model, vae, alphas_cumprod, device, output_dir="KITTI Evaluation"):
    model.eval()
    vae.eval()
    os.makedirs(output_dir, exist_ok=True)

    with open("test_files_eigen.txt", "r") as f:
        lines = f.read().splitlines()

    preprocess = transforms.Compose([
        transforms.Resize((320, 320)),  # 실제 학습 크기에 맞춰 resize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    T = alphas_cumprod.shape[0]

    for idx, rel_path in enumerate(lines):
        abs_path = os.path.join("/media/hspark/My Passport/KITTY dataset", rel_path)
        img = Image.open(abs_path).convert("RGB")
        rgb = preprocess(img).unsqueeze(0).to(device)  # (1, 3, H, W)

        # === VAE 인코딩 ===
        with torch.cuda.amp.autocast():
            image_latent = vae.encode(rgb).latent_dist.sample() * 0.18215

        B = rgb.shape[0]
        H, W = rgb.shape[2:]

        # === 랜덤 노이즈를 depth_latent로 가정 ===
        depth_latent = torch.randn((B, 4, H // 8, W // 8), device=device)  # LDM 크기 기준

        t = torch.full((B,), T - 1, device=device, dtype=torch.long)
        sqrt_alpha = get_sqrt_alpha_t(t, alphas_cumprod)
        sqrt_one_minus_alpha = get_sqrt_one_minus_alpha_t(t, alphas_cumprod)
        noise = torch.randn_like(depth_latent)
        noisy_depth_latent = sqrt_alpha * depth_latent + sqrt_one_minus_alpha * noise

        latent = torch.cat([image_latent, noisy_depth_latent], dim=1)  # (B, 8, h, w)

        # === Multi-step denoising (4-step) ===
        for ratio in [1.0, 0.66, 0.33, 0.0]:
            t_scaled = (t.float() * ratio).round().long()
            pred_z0 = model(latent, t_scaled)
            latent[:, 4:] = pred_z0  # 이미지 latent는 그대로 유지

        pred_depth_latent = latent[:, 4:]  # (B, 4, h, w)
        pred_depth = vae.decode(pred_depth_latent / 0.18215).sample
        pred_depth_first = pred_depth[:, 0:1]  # (B, 1, H, W)
        print("VAE 디코더 출력 (pred depth latent → image):", pred_depth_first.min(), pred_depth_first.max())

        MAX_DEPTH = 80.0  # KITTI Eigen 평가 한계값
        # 1) [-1, 1] 또는 [0, 1] 범위를 '미터' 로 변환 -----------------
        # ─────────────────────────────────────────────────────────────
        #   - 학습 단계에서 디코더 target ⊂ [-1,1] 로 맞췄다면:
        # pred_m = (pred_depth_first + 1) * 0.5 * MAX_DEPTH  # [-1,1] ➔ [0,80]
        #   - 디코더 target ⊂ [0,1] 로 맞췄다면(기존 방식):
        pred_m = pred_depth_first * MAX_DEPTH

        # -------- 평가·시각화 모두 pred_m(미터) 로 진행 -------------
        pred_np = pred_m[0, 0].cpu().numpy()  # (H, W), 단위=m

        # 2) 컬러맵용 선형 정규화 (0m=빨강, 80m=보라) -----------------
        #    percentile→선형 치환으로 '매번 스케일 바뀌는 문제' 제거
        pred_norm = np.clip(pred_np / MAX_DEPTH, 0, 1)  # 0~1
        pred_color = plt.get_cmap("turbo")(1.0 - pred_norm)[..., :3]  # 가까울수록 빨강
        pred_color = (pred_color * 255).astype(np.uint8)

        # 3) 결과 저장 ------------------------------------------------
        # ⚫ 미터 값 자체를 16-bit PNG로 보관하고 싶다면(선택)
        depth_path = os.path.join(output_dir, f"{idx:05d}_depth.png")
        Image.fromarray((pred_np * 256).astype(np.uint16)).save(depth_path)

        # ⚫ 시각화 컬러 PNG
        color_path = os.path.join(output_dir, f"{idx:05d}.png")
        Image.fromarray(pred_color).save(color_path)

        print(f"[{idx + 1}/{len(lines)}] 저장 완료: {color_path}")

        # # === 정규화 및 저장 ===
        # pred_np = pred_depth_first[0, 0].cpu().numpy()
        # valid = pred_np > 0
        # if valid.any():
        #     vmin, vmax = np.percentile(pred_np[valid], 2), np.percentile(pred_np[valid], 98)
        # else:
        #     vmin, vmax = 0.0, 1.0
        # pred_norm = np.clip((pred_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
        # pred_color = plt.get_cmap("magma")(1.0 - pred_norm)[..., :3]  # 먼 거리 어둡게
        # pred_color = (pred_color * 255).astype(np.uint8)
        #
        #
        # out_path = os.path.join(output_dir, f"{idx:05d}.png")
        # Image.fromarray(pred_color).save(out_path)
        #
        # print(f"[{idx+1}/{len(lines)}] 저장 완료: {out_path}")


@torch.no_grad()
def eval_on_nyu_test_set(model, vae, alphas_cumprod, device, output_dir="NYU v2 Evaluation"):
    import glob

    model.eval()
    vae.eval()
    os.makedirs(output_dir, exist_ok=True)

    # 입력 이미지 경로
    nyu_rgb_dir = "/media/hspark/My Passport2/NYU v2 test/nyu_test_rgb/"
    image_paths = sorted(glob.glob(os.path.join(nyu_rgb_dir, "nyu_rgb_*.png")))

    preprocess = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    T = alphas_cumprod.shape[0]

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        rgb = preprocess(img).unsqueeze(0).to(device)

        with torch.cuda.amp.autocast():
            image_latent = vae.encode(rgb).latent_dist.sample() * 0.18215

        B, _, H, W = rgb.shape
        depth_latent = torch.randn((B, 4, H // 8, W // 8), device=device)

        t = torch.full((B,), T - 1, device=device, dtype=torch.long)
        sqrt_alpha = get_sqrt_alpha_t(t, alphas_cumprod)
        sqrt_one_minus_alpha = get_sqrt_one_minus_alpha_t(t, alphas_cumprod)
        noise = torch.randn_like(depth_latent)
        noisy_depth_latent = sqrt_alpha * depth_latent + sqrt_one_minus_alpha * noise

        latent = torch.cat([image_latent, noisy_depth_latent], dim=1)

        for ratio in [1.0, 0.66, 0.33, 0.0]:
            t_scaled = (t.float() * ratio).round().long()
            pred_z0 = model(latent, t_scaled)
            latent[:, 4:] = pred_z0

        pred_depth_latent = latent[:, 4:]
        pred_depth = vae.decode(pred_depth_latent / 0.18215).sample
        pred_depth_first = pred_depth[:, 0:1]
        print("VAE 디코더 출력 (pred depth latent → image):", pred_depth_first.min(), pred_depth_first.max())

        pred_np = pred_depth_first[0, 0].cpu().numpy()
        # valid = pred_np > 0
        # if valid.any():
        #     vmin, vmax = np.percentile(pred_np[valid], 2), np.percentile(pred_np[valid], 98)
        # else:
        #     vmin, vmax = 0.0, 1.0

        # pred_norm = np.clip((pred_np - vmin) / (vmax - vmin + 1e-6), 0, 1)

        # pred_color = plt.get_cmap("turbo")(1.0 - pred_norm)[..., :3]
        pred_color = plt.get_cmap("turbo")(1.0 - pred_np)[..., :3]  # 가까운 거 밝게 보려면 1.0 - depth
        pred_color = (pred_color * 255).astype(np.uint8)

        out_path = os.path.join(output_dir, Path(img_path).name)
        Image.fromarray(pred_color).save(out_path)

        print(f"[{idx+1}/{len(image_paths)}] 저장 완료: {out_path}")


def parse_calib_cam_to_cam(path):
    """Parse P_rect_02 and R_rect_00 from calib_cam_to_cam.txt"""
    P_rect_02, R_rect_00 = None, None
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("P_rect_02:"):
                values = list(map(float, line.strip().split()[1:]))
                P_rect_02 = np.array(values).reshape(3, 4)
            elif line.startswith("R_rect_00:"):
                values = list(map(float, line.strip().split()[1:]))
                R_rect_00 = np.array(values).reshape(3, 3)
    return P_rect_02, R_rect_00

def parse_calib_velo_to_cam(path):
    """Parse Tr_velo_to_cam from calib_velo_to_cam.txt"""
    R, T = None, None
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("R:"):
                R = np.array(list(map(float, line.strip().split()[1:]))).reshape(3, 3)
            elif line.startswith("T:"):
                T = np.array(list(map(float, line.strip().split()[1:]))).reshape(3, 1)
    if R is not None and T is not None:
        return np.concatenate([R, T], axis=1)  # (3, 4)
    raise ValueError("Missing R or T in calib_velo_to_cam.txt")


def project_lidar_to_depth_map_separated_calib(
    bin_path, calib_cam_to_cam_path, calib_velo_to_cam_path, image_shape=(375, 1242)
):
    """Project .bin to 2D depth map using separate calibration files."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    points = points[points[:, 0] > 0]  # Forward-facing only

    P_rect_02, R_rect_00 = parse_calib_cam_to_cam(calib_cam_to_cam_path)
    Tr_velo_to_cam = parse_calib_velo_to_cam(calib_velo_to_cam_path)

    pts_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    pts_cam = R_rect_00 @ (Tr_velo_to_cam @ pts_hom.T)

    pts_2d = P_rect_02 @ np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])
    pts_2d[:2] /= pts_2d[2]

    u, v, z = pts_2d[0], pts_2d[1], pts_cam[2]
    valid = (u >= 0) & (u < image_shape[1]) & (v >= 0) & (v < image_shape[0]) & (z > 0)
    u, v, z = u[valid], v[valid], z[valid]
    u = np.clip(np.round(u).astype(np.int32), 0, image_shape[1] - 1)
    v = np.clip(np.round(v).astype(np.int32), 0, image_shape[0] - 1)

    depth_map = np.zeros(image_shape, dtype=np.float32)
    for i in range(len(z)):
        if depth_map[v[i], u[i]] == 0 or z[i] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = z[i]
    return depth_map


def align_depth_affine(pred, gt, mask):
    """Find scale (a) and shift (b) so that a * pred + b ≈ gt"""
    x = pred[mask].flatten()
    y = gt[mask].flatten()
    A = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a * pred + b


def compute_metrics(pred_aligned, gt, mask):
    """Compute standard depth evaluation metrics between prediction and ground truth."""
    pred = pred_aligned[mask]
    gt = gt[mask]

    # 유효한 로그 연산 범위 마스킹
    valid_log = (pred > 0) & (gt > 0)

    # Accuracy thresholds
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    # Standard error metrics
    abs_rel = np.mean(np.abs(pred - gt) / gt)
    sq_rel = np.mean(((pred - gt) ** 2) / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    # Safe RMSE log
    if valid_log.any():
        rmse_log = np.sqrt(np.mean((np.log(pred[valid_log]) - np.log(gt[valid_log])) ** 2))
    else:
        rmse_log = float('nan')

    return {
        "AbsRel": abs_rel,
        "SqRel": sq_rel,
        "RMSE": rmse,
        "RMSE_log": rmse_log,
        "δ1": a1,
        "δ2": a2,
        "δ3": a3
    }



def evaluate_kitti_predictions(
    pred_list, img_path_list, calib_cam_to_cam_path, calib_velo_to_cam_path
):
    from PIL import Image

    all_metrics = []

    for pred_path, img_path in zip(pred_list, img_path_list):
        pred = np.array(Image.open(pred_path).convert("L")).astype(np.float32) / 255.0
        bin_path = img_path.replace("image_02/data", "velodyne_points/data").replace(".png", ".bin")

        gt = project_lidar_to_depth_map_separated_calib(
            bin_path, calib_cam_to_cam_path, calib_velo_to_cam_path, image_shape=pred.shape
        )
        mask = (gt > 1.0) & (gt < 80.0)

        pred_aligned = align_depth_affine(pred, gt, mask)
        metrics = compute_metrics(pred_aligned, gt, mask)
        all_metrics.append(metrics)

    return {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0]
    }


def main(rank, world_size):

    mode = "eval" # train 또는 eval
    # 1. DDP 초기화 ─────────────────────────────
    setup_ddp(rank, world_size)

    # 2. 프로젝트 경로
    paths = ProjectPaths("/media/hspark/My Passport/LCMDepthv0.2")
    if rank == 0:
        print(f"[Rank {rank}] 프로젝트 디렉토리 초기화 완료")

    # 3. HF 로그인 (rank0만)
    if rank == 0:
        huggingface_login("hf_***")

    # 4. 파이프라인 로드 (아직 DDP X) ───────────
    vae, unet = load_pipeline(rank)
    if unet.conv_in.in_channels == 4:          # 중복 확장 방지
        unet = expand_unet_input_channels(unet, 8)

    alphas_cumprod = get_alphas_cumprod().to(rank).half()  # FP16으로 변환
    model = ConsistencyModel(
        unet=unet,
        alphas_cumprod=alphas_cumprod,
        use_gradient_checkpointing=True,
    ).to(rank).half()  # FP16으로 변환

    # 5. 체크포인트 로드 ───────────────────────────────
    start_epoch, best_loss = 0, float("inf")
    # latest = find_latest_checkpoint(paths.checkpoint_subdirs["latest"])
    if mode == "train":
        ckpt_path = find_latest_checkpoint(paths.checkpoint_subdirs["latest"])
    elif mode == "eval":
        ckpt_path = find_latest_checkpoint(paths.checkpoint_subdirs["best"])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # if latest:
    if ckpt_path:
        # (1) CPU 로 먼저 불러와서 GPU 메모리 세이브
        # ckpt_cpu = torch.load(latest, map_location="cpu")
        ckpt_cpu = torch.load(ckpt_path, map_location="cpu")

        state = ckpt_cpu["model_state_dict"]
        # (2) "module." 접두사 감지 & 제거
        if not any(state_key.startswith("module.") for state_key in model.state_dict().keys()):
            # 현재 모델은 비-DDP → prefix 제거
            state = {k.replace("module.", ""): v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)
        start_epoch = ckpt_cpu["epoch"] + 1
        best_loss = ckpt_cpu.get("loss", best_loss)
        if rank == 0:
            print(f"[Rank {rank}] 체크포인트 로드 완료: {ckpt_path}, epoch {start_epoch}")
            print(f"  ▸ missing={len(missing)}, unexpected={len(unexpected)}")

        # (3) 메모리 정리
        del ckpt_cpu, state
        torch.cuda.empty_cache()
    else:
        if rank == 0:
            print(f"[Rank {rank}] 체크포인트 없음. mode={mode}")

    # 6. DDP 래핑 (마지막 단계) ───────────────────
    model = wrap_model_ddp(model, rank)

    # 7. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)

    if mode == "train":
        # 8. DataLoader
        # dataloader = prepare_dataloader(rank, world_size, width, height)
        dataloader = prepare_dataloader_hypersim(rank, world_size, width, height) # 주석처리할 것
        # data_loader = prepare_dataloader_synscapes_hypersim(
        #     rank=dist.get_rank(),
        #     world_size=dist.get_world_size(),
        #     batch_size=1,              # 필요하면 조정
        #     num_workers=4,
        # )

        # 9. 학습 루프 ───────────────────────────────
        output_dir = "./output/samples"; os.makedirs(output_dir, exist_ok=True)
        num_epochs = 100
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            avg_loss = train_lcm_one_epoch(
                model, vae, optimizer, dataloader, rank,
                alphas_cumprod, epoch, output_dir,
                accumulation_steps=16, save_interval=500
            )

            # ---------- rank 0 저장 ----------
            if rank == 0:
                torch.save(
                    {"epoch": epoch, "loss": avg_loss,
                     "model_state_dict": model.module.state_dict()},
                    os.path.join(paths.checkpoint_subdirs["latest"], f"epoch_{epoch}.pth")
                )
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(
                        {"epoch": epoch, "loss": avg_loss,
                         "model_state_dict": model.module.state_dict()},
                        os.path.join(paths.checkpoint_subdirs["best"],
                                     f"best_epoch_{epoch}_{avg_loss:.6f}.pth")
                    )
                print(f"[Epoch {epoch}] Average Loss: {avg_loss:.6f}")

    elif mode == "eval":
        if rank == 0:
            # KITTI 평가
            eval_on_kitti_test_list(
                model=model,
                vae=vae,
                alphas_cumprod=alphas_cumprod,
                device=rank,
                output_dir="KITTI Evaluation"
            )

            metrics = evaluate_kitti_predictions(
                pred_list=sorted(glob("KITTI Evaluation/*.png")),
                img_path_list=[
                    os.path.join("/media/hspark/My Passport/KITTY dataset", line.strip())
                    for line in open("test_files_eigen.txt")
                ],
                calib_cam_to_cam_path="./calib_cam_to_cam.txt",
                calib_velo_to_cam_path="./calib_velo_to_cam.txt"
            )

            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            print("📊 KITTI 평가 결과:")
            for k, v in metrics.items():
                print(f" - {k}: {v:.4f}")

            # NYU v2 평가
            # eval_on_nyu_test_set(
            #     model=model,
            #     vae=vae,
            #     alphas_cumprod=alphas_cumprod,
            #     device=rank,
            #     output_dir="NYU v2 Evaluation"
            # )

    # 10. 종료 ──────────────────────────────────
    dist.barrier(device_ids=[rank])
    cleanup_ddp()
    print(f"[Rank {rank}] 작업 완료")


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"사용 가능한 GPU 수: {world_size}")
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)


