import os
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
from core.utils.utils import InputPadder
from core.monster import Monster 
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets as datasets
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys
sys.path.append("/mnt/datalsj/depth/Depth-Anything-V2/metric_depth/dataset")
from mono import mono



def plot_and_save_results(image, disp_pred, disp_pred_final,disp_gt, depth_gt, save_path="results.png"):
    import matplotlib.pyplot as plt
    
    # 将tensor转换为numpy数组并调整维度顺序
    image_np = image[0][0].cpu().detach().squeeze().numpy()  # 只取第一通道
    disp_pred_np = disp_pred[0].cpu().detach().squeeze().numpy()
    disp_gt_np = disp_gt[0].cpu().detach().squeeze().numpy()
    depth_gt_np = depth_gt[0].cpu().detach().squeeze().numpy()
    disp_pred_final_np = disp_pred_final[0].cpu().detach().squeeze().numpy()
   # 计算误差图
    error_map = np.abs(disp_pred_np - disp_gt_np)
    
    # 计算各图的均值
    disp_pred_mean = np.mean(disp_pred_np)
    disp_pred_final_mean = np.mean(disp_pred_final_np)
    disp_gt_mean = np.mean(disp_gt_np)
    error_mean = np.mean(error_map)
    
    # 创建一个大图
    plt.figure(figsize=(20, 5))
    
    # 绘制输入图像（灰度图）
    plt.subplot(1, 6, 1)
    plt.imshow(image_np, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    
    # 绘制预测视差图（红白蓝）
    plt.subplot(1, 6, 2)
    disp_pred_plot = plt.imshow(disp_pred_np, cmap='bwr', vmin=np.min(disp_pred_np), vmax=np.max(disp_pred_np))
    plt.title(f"Predicted Disparity\nMean: {disp_pred_mean:.2f}")
    plt.axis('off')
    plt.colorbar(disp_pred_plot, fraction=0.046, pad=0.04)
    
    plt.subplot(1, 6, 3)
    disp_pred_plot = plt.imshow(disp_pred_final_np, cmap='bwr', vmin=np.min(disp_pred_final_np), vmax=np.max(disp_pred_final_np))
    plt.title(f"Predicted Disparity Final\nMean: {disp_pred_final_mean:.2f}")
    plt.axis('off')
    plt.colorbar(disp_pred_plot, fraction=0.046, pad=0.04)

    # 绘制真实视差图（红白蓝）
    plt.subplot(1, 6, 4)
    disp_gt_plot = plt.imshow(disp_gt_np, cmap='bwr', vmin=np.min(disp_gt_np), vmax=np.max(disp_gt_np))
    plt.title(f"Ground Truth Disparity\nMean: {disp_gt_mean:.2f}")
    plt.axis('off')
    plt.colorbar(disp_gt_plot, fraction=0.046, pad=0.04)
    
    # 绘制误差图
    plt.subplot(1, 6, 5)
    error_plot = plt.imshow(error_map, cmap='hot')
    plt.title(f"Error Map\nMean: {error_mean:.2f}")
    plt.axis('off')
    plt.colorbar(error_plot, fraction=0.046, pad=0.04)
    
    # 绘制真实深度图
    plt.subplot(1, 6, 6)
    depth_plot = plt.imshow(depth_gt_np, cmap='plasma')
    plt.title("Ground Truth Depth")
    plt.axis('off')
    plt.colorbar(depth_plot, fraction=0.046, pad=0.04)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()




def gray_2_colormap_np(img, cmap = 'rainbow', max = None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img/(max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.colormaps[cmap]
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap

def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    # valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # quantile = torch.quantile((disp_init_pred - disp_gt).abs(), 0.9)
    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)#  & ((disp_init_pred - disp_gt).abs() < quantile)
    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        # quantile = torch.quantile(i_loss, 0.9)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        'train/epe': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean(),
    }
    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    DPT_params = list(map(id, model.feat_decoder.parameters())) 
    rest_params = filter(lambda x:id(x) not in DPT_params and x.requires_grad, model.parameters())

    params_dict = [{'params': model.feat_decoder.parameters(), 'lr': args.lr/2.0}, 
                   {'params': rest_params, 'lr': args.lr}, ]
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr/2.0, args.lr], args.total_step+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')


    return optimizer, scheduler

@hydra.main(version_base=None, config_path='config', config_name='train_mono')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='bf16', dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    
    # 初始化 TensorBoard
    if accelerator.is_main_process:
        log_dir = Path(cfg.save_path) / 'tensorboard_logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=str(log_dir))
    else:
        writer = None

    # train_dataset = datasets.fetch_dataloader(cfg)
    train_dataset = mono(mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size//cfg.num_gpu,
        pin_memory=True, shuffle=True, num_workers=int(4), drop_last=True)

    aug_params = {}
    # val_dataset = datasets.KITTI(aug_params, image_set='training')
    val_dataset = mono(mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(1),
        pin_memory=True, shuffle=False, num_workers=int(4), drop_last=False)
    model = Monster(cfg)
    if cfg.restore_ckpt is not None:
        assert cfg.restore_ckpt.endswith(".pth")
        print(f"Loading checkpoint from {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            ckpt[key.replace('module.', '')] = checkpoint[key]
        model.load_state_dict(ckpt, strict=True)
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
    del ckpt, checkpoint
    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(train_loader, model, optimizer, lr_scheduler, val_loader)
    model.to(accelerator.device)

    total_step = 0
    should_keep_training = True
    while should_keep_training:
        active_train_loader = train_loader
        model.train()
        if hasattr(model, 'module'):
            model.module.freeze_bn()
        else:
            model.freeze_bn()
        for data in tqdm(active_train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            _, left, right, disp_gt, valid = [x for x in data]
            with accelerator.autocast():
                disp_init_pred, disp_preds, depth_mono = model(left, right, iters=cfg.train_iters)
            loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, max_disp=cfg.max_disp)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_step += 1
            loss = accelerator.reduce(loss.detach(), reduction='mean')
            metrics = accelerator.reduce(metrics, reduction='mean')
            
            # 使用 TensorBoard 记录日志
            if accelerator.is_main_process and writer is not None:
                writer.add_scalar('train/loss', loss.item(), total_step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], total_step)
                for key, value in metrics.items():
                    writer.add_scalar(key, value.item(), total_step)

            ####visualize the depth_mono and disp_preds
            if total_step % 20 == 0 and accelerator.is_main_process:
                image1_np = left[0].squeeze().cpu().numpy()
                image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                image1_np = image1_np.astype(np.uint8)
                image1_np = np.transpose(image1_np, (1, 2, 0))

                image2_np = right[0].squeeze().cpu().numpy()
                image2_np = (image2_np - image2_np.min()) / (image2_np.max() - image2_np.min()) * 255.0
                image2_np = image2_np.astype(np.uint8)
                image2_np = np.transpose(image2_np, (1, 2, 0))


                depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze())
                disp_preds_np = gray_2_colormap_np(disp_preds[-1][0].squeeze())
                disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())
                
                # 使用 TensorBoard 记录图像
                if writer is not None:
                    writer.add_image("disp_pred", disp_preds_np.transpose(2, 0, 1), total_step)
                    writer.add_image("disp_gt", disp_gt_np.transpose(2, 0, 1), total_step)
                    writer.add_image("depth_mono", depth_mono_np.transpose(2, 0, 1), total_step)
            

            if total_step % 100 == 0 and accelerator.is_main_process:
                # 创建保存目录
                os.makedirs(os.path.join('/mnt/datalrl/depth/stereo/MonSter/plot', 'train'), exist_ok=True)
                plot_and_save_results(left, disp_init_pred, disp_preds[-1], disp_gt, depth_mono, os.path.join('/mnt/datalrl/depth/stereo/MonSter/plot', 'train', f'{total_step}.png'))




            if (total_step > 0) and (total_step % cfg.save_frequency == 0):
                if accelerator.is_main_process:
                    save_path = Path(cfg.save_path + '/%d.pth' % (total_step))
                    model_save = accelerator.unwrap_model(model)
                    torch.save(model_save.state_dict(), save_path)
                    del model_save
        
            if (total_step > 0) and (total_step % cfg.val_frequency == 0):

                model.eval()
                elem_num, total_epe, total_out = 0, 0, 0
                for data in tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
                    _, left, right, disp_gt, valid = [x for x in data]
                    padder = InputPadder(left.shape, divis_by=32)
                    left, right = padder.pad(left, right)
                    with torch.no_grad():
                        disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                    disp_pred = padder.unpad(disp_pred)
                    assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                    epe = torch.abs(disp_pred - disp_gt)
                    out = (epe > 1.0).float()
                    epe = torch.squeeze(epe, dim=1)
                    out = torch.squeeze(out, dim=1)
                    epe, out = accelerator.gather_for_metrics((epe[valid >= 0.5].mean(), out[valid >= 0.5].mean()))
                    elem_num += epe.shape[0]
                    for i in range(epe.shape[0]):
                        total_epe += epe[i]
                        total_out += out[i]
                    
                    # 使用 TensorBoard 记录验证指标
                    if accelerator.is_main_process and writer is not None:
                        writer.add_scalar('val/epe', total_epe / elem_num, total_step)
                        writer.add_scalar('val/d1', 100 * total_out / elem_num, total_step)

                model.train()
                if hasattr(model, 'module'):
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()

            if total_step == cfg.total_step:
                should_keep_training = False
                break

    if accelerator.is_main_process:
        save_path = Path(cfg.save_path + '/final.pth')
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path)
        del model_save
        
        # 关闭 TensorBoard writer
        if writer is not None:
            writer.close()
    
    accelerator.end_training()

if __name__ == '__main__':
    main()