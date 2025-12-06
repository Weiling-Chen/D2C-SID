import os
import torch
from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import kl
from warmup_scheduler import GradualWarmupScheduler


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _train(model, args):
    # 设置随机数种子
    setup_seed(3407)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-7)
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    warmup_epochs = 3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch - warmup_epochs,
                                                                  eta_min=1e-6)
    # scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        print('Resume from %d' % epoch)
        epoch += 1

    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    epoch_kl_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    iter_kl_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr = -1

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img, mu_po, logvar_po, mu_pr, logvar_pr, dist_po, dist_pr = model(input_img, label_img, training=True)
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1 + l2 + l3

            loss_kl = torch.mean(kl.kl_divergence(dist_po, dist_pr))

            label_fft1 = torch.fft.fft2(label_img4, dim=(-2, -1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

            pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2, -1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

            label_fft2 = torch.fft.fft2(label_img2, dim=(-2, -1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

            pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2, -1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

            label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

            pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2, -1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1 + f2 + f3

            # loss = loss_content + 0.1 * loss_fft  # 原始没有vae

            loss = loss_content + 0.1 * loss_fft + 0.01 * loss_kl  # 采样10 lr1e-5 bz 16
            # loss = loss_content + 0.1*loss_fft + 0.01*loss_kl  # 采样10 lr1e-5 bz 16
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())
            iter_kl_adder(loss_kl.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())
            epoch_kl_adder(loss_kl.item())

            if (iter_idx + 1) % args.print_freq == 0:
                print(
                    "Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss_content: %7.4f Loss_fft: %7.4f Loss_kl: %.4f" % (
                        iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0],
                        iter_pixel_adder.average(),
                        iter_fft_adder.average(),
                        iter_kl_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('KL Loss', iter_kl_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)

                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
                iter_kl_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel_Loss: %7.4f Epoch FFT_Loss: %7.4f Epoch KL_Loss: %.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average(),
            epoch_kl_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        epoch_kl_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                best_psnr = val
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    print("best_psnr%.3f" % (best_psnr))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
