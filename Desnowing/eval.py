import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
# from pytorch_msssim import ssim
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as f

from skimage import img_as_ubyte
import cv2


def save_img_1(img, img_name, i, out):
    save_img = img.squeeze(dim=0).clamp(0, 1).numpy().transpose(1, 2, 0)
    # save img
    save_dir = os.path.join(r'D:\lfq', out)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name_list = img_name.split('.', 1)
    save_fn = save_dir + '/' + name_list[0] + '_' + str(i) + '.' + name_list[1]
    # cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(save_fn, save_img*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    factor = 4
    total_time = 0.0
    num_images = len(dataloader)
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            tm = time.time()

            pred = model(input_img, label_img, training=False)[2]
            # The average PSNR is 34.16 dB
            # The average SSIM is 0.91873 dB
            avg_pre = 0
            for i in range(10):
                t0 = time.time()
                pred = model(input_img, label_img, training=False)[2]
                t1 = time.time()
                avg_pre = avg_pre + pred / 10
                save_img_1(pred.cpu().data, name[0], i, r'D:\lfq')

                print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            # save_img_2(avg_pre.cpu().data, name[0], opt.reference_out)
            # pred = pred[:,:,:h,:w]
            pred = avg_pre[:,:,:h,:w]

            elapsed = time.time() - tm
            total_time += elapsed
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()


            label_img = (label_img).cuda()
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
            down_ratio = max(1, round(min(H, W) / 256))	
            # ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))),
            #                 f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))),
            #                 data_range=1, size_average=False)
            ssim_val = 0
            # ssim_val = ssim(label_img, denoised, data_range=1.0, multichannel=not args.gray)
            print('%d iter PSNR_dehazing: %.2f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
            ssim_adder(ssim_val)

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'L')
                pred.save(save_name)
            
            psnr_mimo = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr_val)

            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr_mimo, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.5f dB' % (ssim_adder.average()))

        print("Average time: %f" % adder.average())
        aver_time = total_time / num_images
        print(aver_time)

