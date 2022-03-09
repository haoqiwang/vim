#!/usr/bin/env python
import argparse
from mmcls.apis import init_model
import torch
from list_dataset import ImageFilelist
import numpy as np
import pickle
from tqdm import tqdm
import mmcv
from os.path import dirname
import torchvision as tv

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('data_root', help='Path to data')
    parser.add_argument('out_file', help='Path to output file')
    parser.add_argument('--cfg', default='vit-base-p16-384.py', help='Path to config')
    parser.add_argument('--checkpoint', default='checkpoints/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth', help='Path to checkpoint')
    parser.add_argument('--img_list', default=None, help='Path to image list')
    parser.add_argument('--batch', type=int, default=256, help='Path to data')
    parser.add_argument('--workers', type=int, default=4, help='Path to data')
    parser.add_argument('--fc_save_path', default=None, help='Path to save fc')

    return parser.parse_args()


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    cfg = mmcv.Config.fromfile(args.cfg)
    model = init_model(cfg, args.checkpoint, 0).cuda().eval()

    if args.fc_save_path is not None:
        mmcv.mkdir_or_exist(dirname(args.fc_save_path))
        w = model.head.layers.head.weight.cpu().detach().numpy()
        b = model.head.layers.head.bias.cpu().detach().numpy()
        with open(args.fc_save_path, 'wb') as f:
            pickle.dump([w, b], f)
        return

    transform = tv.transforms.Compose([
        tv.transforms.Resize((384, 384)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.img_list is not None:
        dataset = ImageFilelist(args.data_root, args.img_list, transform)
    else:
        dataset = tv.datasets.ImageFolder(args.data_root, transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    features = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x.cuda()
            feat_batch = model.backbone(x)[0][-1].cpu().numpy()
            features.append(feat_batch)

    features = np.concatenate(features, axis=0)

    mmcv.mkdir_or_exist(dirname(args.out_file))
    with open(args.out_file, 'wb') as f:
        pickle.dump(features, f)

if __name__ == '__main__':
    main()
