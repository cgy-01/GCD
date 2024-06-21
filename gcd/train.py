import argparse
import os
import torch.nn as nn
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
from torchvision.utils import save_image

sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.levirloader import LEVIRDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary

seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    # 归一化
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def main():
    args = create_argparser().parse_args()
    th.cuda.set_device(0)
    logger.configure(dir=args.out_dir)

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)
        ds = ISICDataset(args, args.data_dir, transform_test)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size, args.image_size))]
        transform_test = transforms.Compose(tran_list)
        ds = BRATSDataset3D(args.data_dir, transform_test)
        args.in_ch = 5
    elif args.data_name == 'LRVIR-CD':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)
        ds = LEVIRDataset(args, args.data_dir, transform_test)
        args.in_ch = 3

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=True
    )
    data = iter(datal)
    # print(data[0][2])

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # 初始化 GPU 和模型
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.train()  # 设置模型为训练模式

    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)

    time_step = 0
    all_images = []

    while len(all_images) * args.batch_size < args.num_samples:
        A_img, B_img, CD_img = next(data)
        c = th.randn_like(A_img[:, :1, ...])
        sum = th.cat((A_img, B_img), dim=1)
        img = th.cat((sum, c), dim=1)

        time_step += 1
        logger.log("sampling...")

        enslist = []

        for i in range(1):
            model_kwargs = {}
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step=args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            save_image(CD_img, f"./CD/{time_step}.png")
            save_image(cal_out, f"./cal/{time_step}_{i}.png")
            co = cal_out.clone().detach()  # 使用clone().detach()替换torch.tensor()
            enslist.append(co)

        ensres = staple(th.stack(enslist, dim=0)).squeeze(0)
        vutils.save_image(ensres, fp=f"./ens/{time_step}.png", nrow=1, padding=10)

        CD_img = CD_img.to(th.device('cuda'))

        optimizer.zero_grad()
        loss = calculate_loss(cal_out, CD_img)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()


def create_argparser():
    defaults = dict(
        data_name='LRVIR-CD',
        data_dir=r"F:\Datasets\ChangeDetection\LEVIR-CD",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=1,  # number of samples in the ensemble
        gpu_dev="0",
        out_dir='./results_5_25_LEVIR_test/',
        multi_gpu=None,  # "0,1,2"
        debug=False,
        learning_rate=1e-4  # 添加学习率参数
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def calculate_loss(predicted, target):
    criterion = nn.MSELoss()
    loss = criterion(predicted, target)
    return loss


if __name__ == "__main__":
    main()
