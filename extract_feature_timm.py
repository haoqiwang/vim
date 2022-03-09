#!/usr/bin/env python
import argparse
import torch
from list_dataset import ImageFilelist
import numpy as np
import pickle
from tqdm import tqdm
import mmcv
from os.path import dirname
import torchvision as tv
import timm

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('data_root', help='Path to data')
    parser.add_argument('out_file', help='Path to output file')
    parser.add_argument('model', help='Path to config')
    parser.add_argument('--checkpoint', default='checkpoints/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth', help='Path to checkpoint')
    parser.add_argument('--img_list', default=None, help='Path to image list')
    parser.add_argument('--batch', type=int, default=256, help='Path to data')
    parser.add_argument('--workers', type=int, default=4, help='Path to data')
    parser.add_argument('--fc_save_path', default=None, help='Path to save fc')

    return parser.parse_args()


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    if args.fc_save_path is not None:
        model = timm.create_model(args.model, pretrained=True)
        mmcv.mkdir_or_exist(dirname(args.fc_save_path))
        if args.model in ['repvgg_b3']:
            w = model.head.fc.weight.cpu().detach().numpy()
            b = model.head.fc.bias.cpu().detach().numpy()
        elif args.model in ['swin_base_patch4_window7_224', 'deit_base_patch16_224']:
            w = model.head.weight.cpu().detach().numpy()
            b = model.head.bias.cpu().detach().numpy()
        else:
            w = model.fc.weight.cpu().detach().numpy()
            b = model.fc.bias.cpu().detach().numpy()
        with open(args.fc_save_path, 'wb') as f:
            pickle.dump([w, b], f)
        return

    model = timm.create_model(args.model, pretrained=True, num_classes=0).cuda().eval()

    transform = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
            feat_batch = model(x).cpu().numpy()
            features.append(feat_batch)

    features = np.concatenate(features, axis=0)

    mmcv.mkdir_or_exist(dirname(args.out_file))
    with open(args.out_file, 'wb') as f:
        pickle.dump(features, f)

if __name__ == '__main__':
    main()
