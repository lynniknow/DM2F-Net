# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, HAZERD_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model_new import DM2FNet, DM2FNet_woPhy
from datasets import SotsDataset, OHazeDataset, HazeRDDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.metrics import mean_squared_error ##
from skimage import color ##
from colormath.color_objects import sRGBColor, LabColor ##
from colormath.color_conversions import convert_color ##
from colormath.color_diff import delta_e_cie2000 ##



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
#exp_name = 'O-Haze-complex'
# exp_name = 'O-Haze-channel'
# exp_name = 'RESIDE_channel'
# exp_name = 'RESIDE_complex'
exp_name = 'RESIDE_new'

args = {
    'snapshot': 'iter_15000_loss_0.01546_lr_0.000000',
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    # 'O-Haze': OHAZE_ROOT,
    'HazeRD': HAZERD_ROOT,
}

to_pil = transforms.ToPILImage()

def calculate_ciede2000_batch(rgb1, rgb2):
    lab1 = color.rgb2lab(rgb1)
    lab2 = color.rgb2lab(rgb2)
    delta_e = color.deltaE_ciede2000(lab1, lab2)
    return np.mean(delta_e)

def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNet().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNet_woPhy().cuda()
                dataset = OHazeDataset(root, 'O-HAZY NTIRE 2018', 'test')
            elif 'HazeRD' in name:
                net = DM2FNet().cuda()
                dataset = HazeRDDataset(root, ['50','100', '200', '500', '1000'])
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims, ciede2000s, mses = [], [], [], []
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False, channel_axis=2)
                    ssims.append(ssim)
                    ciede2000 = calculate_ciede2000_batch(gt, r)
                    ciede2000s.append(ciede2000)
                    mse = mean_squared_error(gt, r)
                    mses.append(mse)
                    print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}'
                          .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(ciede2000s):.6f}, MSE: {np.mean(mses):.6f}")


if __name__ == '__main__':
    main()