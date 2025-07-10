import torch
import torch.multiprocessing as mp
from pathlib import Path
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from huggingface_hub import login, whoami
from torch.cuda.amp import autocast, GradScaler

from models import (
    DepthEstimationUNet,
    VAEWrapper,
    LCMScheduler,
    LCMSchedulerConfig,
    get_model_output_cfg
)
from data import (
    VirtualKITTI2Dataset,
    HyperSimDataset,
    RatioDistributedSampler,
    KITTIDataset
)
from training import (
    LCMTrainer,
    LCMLoss,
    create_optimizer
)
from utils import (
    setup_ddp,
    cleanup_ddp,
    save_checkpoint,
    load_checkpoint,
    save_depth_prediction,
    compute_depth_metrics,
    project_points_to_image,
    save_samples
)
from configs import (
    ProjectPaths,
    ModelConfig,
    TrainingConfig,
    LoggingConfig,
    InferenceConfig
)

def huggingface_login(token: str):
    """Hugging Face 로그인"""
    login(token)
    user_info = whoami()
    print(f"Logged in as: {user_info['name']}")


def train(
        rank: int,
        world_size: int,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        logging_config: LoggingConfig,
        project_paths: ProjectPaths
):
    """학습 프로세스"""
    # DDP 설정
    setup_ddp(rank, world_size)

    # 데이터셋 설정
    vkitti_dataset = VirtualKITTI2Dataset(
        data_dir="/media/hspark/My Passport/Virtual Kitti 2",
        camera_id=0,
        width=model_config.image_size[0],
        height=model_config.image_size[1]
    )

    hypersim_dataset = HyperSimDataset(
        data_dir="/media/hspark/My Passport2/downloads",
        metadata_csv_path="./metadata_images_split_scene_v1.csv",
        width=model_config.image_size[0],
        height=model_config.image_size[1],
        rank=rank
    )

    # 데이터셋 결합
    combined_dataset = ConcatDataset([vkitti_dataset, hypersim_dataset])

    # Sampler 설정
    train_sampler = RatioDistributedSampler(
        combined_dataset,
        vkitti_size=len(vkitti_dataset),
        hypersim_size=len(hypersim_dataset),
        num_replicas=world_size,
        rank=rank
    )

    # DataLoader 설정
    train_loader = DataLoader(
        combined_dataset,
        batch_size=train_config.batch_size,
        sampler=train_sampler,
        num_workers=train_config.num_workers,
        pin_memory=True
    )

    # 모델 설정
    if rank == 0:
        print("Initializing models...")

    # Base 모델 로드
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=torch.float16,  # FP16 for memory efficiency
        use_safetensors=True
    )

    # VAE 설정 - pipeline의 VAE 직접 사용
    vae = VAEWrapper(pipeline.vae).to(rank).half()  # FP16

    # UNet 설정
    unet = DepthEstimationUNet(pipeline.unet).to(rank).half()  # FP16

    # DDP 래핑
    unet = torch.nn.parallel.DistributedDataParallel(
        unet,
        device_ids=[rank]
    )

    # LCM 컴포넌트 설정
    scheduler_config = LCMSchedulerConfig(
        # num_inference_steps=model_config.num_inference_steps,
        num_timesteps=model_config.num_train_timesteps,
        sigma_min=model_config.sigma_min,
        sigma_max=model_config.sigma_max,
        rho=model_config.rho
    )
    scheduler = LCMScheduler(scheduler_config)

    # Loss 및 Optimizer 설정
    loss_fn = LCMLoss(scheduler).to(rank)
    optimizer = create_optimizer(unet.parameters(), train_config.optimizer)

    # Trainer 초기화
    trainer = LCMTrainer(
        model=unet,
        vae=vae,
        scheduler=scheduler,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=rank,
        rank=rank,
        config=train_config,
        logging_config=logging_config,
        train_dataloader=train_loader
    )

    # ✅ train_dataloader 확인
    if trainer.train_dataloader is None:
        raise ValueError("❌ train_dataloader가 None입니다. 올바르게 전달되지 않았습니다!")
    else:
        print(f"✅ train_dataloader가 정상적으로 존재합니다. 총 배치 개수: {len(trainer.train_dataloader)}")

    # # ✅ 강제로 `save_samples()` 실행하여 `AttributeError`가 발생하는지 확인
    # print("🔍 save_samples() 테스트 실행...")
    # save_samples(trainer, project_paths.sample_paths['training'], epoch=0)
    # print("✅ save_samples() 실행 완료, train_dataloader 문제 없음!")

    # 체크포인트 로드 (있는 경우)
    start_epoch = 0
    # if checkpoint_path := list(project_paths.checkpoint_paths['latest'].glob("*.pth")):
    #     checkpoint = load_checkpoint(
    #         str(checkpoint_path[0]),
    #         unet,
    #         optimizer,
    #         map_location=f'cuda:{rank}'
    #     )
    #     start_epoch = checkpoint['epoch'] + 1
    latest_ckpts = list(project_paths.checkpoint_paths['latest'].glob("*.pth"))
    if latest_ckpts:
        # 파일 이름에서 에폭 번호를 추출하여 내림차순 정렬
        latest_ckpts.sort(key=lambda x: int(x.stem.split('_')[-1]), reverse=True)
        checkpoint = load_checkpoint(
            str(latest_ckpts[0]),
            unet,
            optimizer,
            map_location=f'cuda:{rank}'
        )
        start_epoch = checkpoint['epoch'] + 1

    # 학습 루프
    for epoch in range(start_epoch, train_config.num_epochs):
        train_sampler.set_epoch(epoch)

        # 한 에폭 학습
        epoch_loss = trainer.train_epoch(train_loader, epoch)

        # 체크포인트 저장 (rank 0에서만)
        if rank == 0:
            # 주기적인 로깅
            if (epoch + 1) % logging_config.log_every_n_steps == 0:
                print(f"Epoch {epoch + 1}/{train_config.num_epochs}, Loss: {epoch_loss:.4f}")

            # # 샘플 저장
            # if (epoch + 1) % logging_config.save_sample_every_n_epochs == 0:
            #     # 샘플 생성 및 저장 로직
            #     save_samples(trainer, project_paths.sample_paths['training'], epoch)

            # 주기적으로 모델 저장
            if (epoch + 1) % train_config.save_every_n_epochs == 0:
                save_checkpoint(
                    project_paths.checkpoint_paths['periodic'],
                    'periodic',
                    epoch,
                    unet,
                    optimizer,
                    epoch_loss
                )

            # Best 모델 저장
            if epoch_loss < trainer.best_loss:
                trainer.best_loss = epoch_loss
                save_checkpoint(
                    project_paths.checkpoint_paths['best'],
                    'best',
                    epoch,
                    unet,
                    optimizer,
                    epoch_loss
                )

            # Latest 체크포인트 저장
            save_checkpoint(
                project_paths.checkpoint_paths['latest'],
                'latest',
                epoch,
                unet,
                optimizer,
                epoch_loss
            )

    cleanup_ddp()


def evaluate(
        rank: int,
        world_size: int,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
        project_paths: ProjectPaths
):
    """평가 프로세스"""
    # DDP 설정
    setup_ddp(rank, world_size)

    # 데이터셋 경로 설정
    data_dir = Path("/media/hspark/My Passport/KITTY dataset")
    test_file_path = "test_files_eigen.txt"

    if rank == 0:
        # 출력 디렉토리 생성
        predictions_dir = project_paths.eval_paths['predictions']
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # KITTI 데이터셋 검증
        print("\nValidating KITTI dataset paths...")
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        if not Path(test_file_path).exists():
            raise FileNotFoundError(f"Test file list not found: {test_file_path}")

    # 모델 설정
    if rank == 0:
        print("Initializing models...")

    # Base 모델 로드
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=torch.float16,  # FP16 for memory efficiency
        use_safetensors=True
    )

    # VAE 설정
    vae = VAEWrapper(pipeline.vae).to(rank).half()  # FP16

    # UNet 설정
    unet = DepthEstimationUNet(pipeline.unet).to(rank).half()  # FP16

    # DDP 래핑
    unet = torch.nn.parallel.DistributedDataParallel(
        unet,
        device_ids=[rank]
    )

    # LCM 스케줄러 설정
    scheduler_config = LCMSchedulerConfig(
        num_inference_steps=inference_config.num_inference_steps,
        sigma_min=model_config.sigma_min,
        sigma_max=model_config.sigma_max,
        rho=model_config.rho
    )
    scheduler = LCMScheduler(scheduler_config)

    # 최적의 모델 체크포인트 로드
    checkpoint_paths = list(project_paths.checkpoint_paths['best'].glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoint found in best model directory")

    best_checkpoint_path = min(
        checkpoint_paths,
        key=lambda x: float(str(x).split('loss_')[-1].replace('.pth', ''))
    )

    if rank == 0:
        print(f"Loading checkpoint: {best_checkpoint_path}")

    load_checkpoint(str(best_checkpoint_path), unet, map_location=f'cuda:{rank}')

    # 평가 데이터셋 설정
    dataset = KITTIDataset(
        data_dir=data_dir,
        test_file_path=test_file_path,
        width=model_config.image_size[0],
        height=model_config.image_size[1]
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=inference_config.batch_size,
        sampler=sampler,
        num_workers=4
    )

    # 평가 실행
    unet.eval()
    metrics_all = {
        'abs_rel': [], 'sq_rel': [], 'rmse': [], 'rmse_log': [],
        'delta1': [], 'delta2': [], 'delta3': []
    }

    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="Evaluating")

    with torch.no_grad(), autocast(dtype=torch.float16):
        for batch in dataloader:
            # 이미지를 디바이스로 이동
            rgb = batch['rgb'].to(rank)

            # Latent 인코딩
            image_latent = vae.encode(rgb)

            # 초기 노이즈
            batch_size = rgb.shape[0]
            depth_latent = torch.randn(
                (batch_size, 4, image_latent.shape[2], image_latent.shape[3]),
                device=rank
            )

            # LCM 샘플링
            # for t in range(scheduler.config.num_inference_steps):
            #     # 노이즈 예측
            #     noise_pred = unet(
            #         torch.cat([image_latent, depth_latent], dim=1),
            #         torch.full((batch_size,), t, device=rank, dtype=torch.long)
            #     )
            #
            #     # 다음 샘플 계산
            #     depth_latent = scheduler.step(noise_pred, t, depth_latent)

            # 전체 타임스텝에서 내림차순으로 스텝 선택 (예: 1000 → 0)
            timesteps = torch.linspace(inference_config.num_timesteps - 1, 0,
                                       steps=inference_config.num_inference_steps,
                                       dtype=torch.long).to(rank)

            for t in timesteps:
                t_batch = torch.full((batch_size,), t.item(), device=rank, dtype=torch.long)
                # CFG 적용된 노이즈 예측 (조건부와 무조건부 입력을 get_model_output_cfg 함수에서 생성)
                noise_pred = get_model_output_cfg(
                    model=unet,
                    latents=depth_latent,
                    image_latents=image_latent,
                    t=t_batch,
                    guidance_scale=inference_config.guidance_scale
                )
                # DDIM 업데이트를 통한 이전 샘플 생성
                depth_latent = scheduler.step(noise_pred, t.item(), depth_latent)

            # 깊이맵 디코딩
            depth_images = vae.decode(depth_latent)

            # 예측 저장 및 평가
            for i, depth_img in enumerate(depth_images):
                depth_map = depth_img[0].cpu().numpy()

                if rank == 0:
                    # 예측 저장
                    output_path = predictions_dir / f"{batch['file_name'][i]}"
                    save_depth_prediction(depth_map, str(output_path))

                    # # Ground truth 로드 및 메트릭 계산
                    # gt_path = data_dir / batch['gt_path'][i]
                    # gt_points = np.fromfile(gt_path, dtype=np.float32).reshape(-1, 4)
                    #
                    # # 수정: 각 샘플의 calibration 정보를 추출
                    # calib = {key: batch['calibration'][key][i] for key in batch['calibration']}
                    #
                    # # GT 포인트를 이미지 평면에 투영
                    # gt_points_img, gt_depth = project_points_to_image(
                    #     gt_points[:, :3],
                    #     calib['P_rect'],
                    #     calib['R_rect'],
                    #     calib['T_velo_cam'],  # 만약 calibration에서 'T_velo_cam' 대신 'V2C'를 사용한다면, 적절히 변환해야 합니다.
                    #     (model_config.image_size[1], model_config.image_size[0])
                    # )
                    #
                    # # 메트릭 계산
                    # metrics = compute_depth_metrics(depth_map, gt_depth)
                    # for k, v in metrics.items():
                    #     metrics_all[k].append(v)

            if rank == 0:
                pbar.update(1)

    if rank == 0:
        pbar.close()

        # 최종 메트릭 계산 및 저장
        final_metrics = {k: np.mean(v) for k, v in metrics_all.items()}

        # 결과 출력
        print("\nEvaluation Results:")
        for k, v in final_metrics.items():
            print(f"{k}: {v:.3f}")

        # 결과 저장
        results_file = project_paths.eval_paths['metrics'] / "evaluation_results.txt"
        with open(results_file, 'w') as f:
            for k, v in final_metrics.items():
                f.write(f"{k}: {v:.6f}\n")

    cleanup_ddp()


def main():
    """메인 실행 함수"""
    # 설정 초기화
    model_config = ModelConfig()
    train_config = TrainingConfig()
    inference_config = InferenceConfig()
    logging_config = LoggingConfig()
    project_paths = ProjectPaths()

    # HuggingFace 로그인
    huggingface_login("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")

    # GPU 수 확인
    world_size = torch.cuda.device_count()

    # 모드 선택
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval'])
    args = parser.parse_args()
    args.mode = 'eval'

    # 선택된 모드로 실행
    if args.mode == 'train':
        mp.spawn(
            train,
            args=(world_size, model_config, train_config,
                  logging_config, project_paths),
            nprocs=world_size,
            join=True
        )
    else:
        mp.spawn(
            evaluate,
            args=(world_size, model_config, inference_config,
                  project_paths),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()