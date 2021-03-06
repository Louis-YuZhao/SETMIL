"""
Extract feature offline
"""
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import cv2
import rich
import queue
import random
import pickle
import argparse
import pandas as pd
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm
from PIL import Image
from rich import progress
from itertools import repeat, cycle
import torch.multiprocessing as mp
import threading
from multiprocessing.pool import ThreadPool, Pool
from concurrent.futures import ThreadPoolExecutor
from efficientnet_pytorch import EfficientNet
from configs import get_cfg_defaults, update_default_cfg

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

import torch.utils.data as data_utils
import pretrainedmodels
import torchvision
import torchvision.transforms as transforms

import albumentations as alb
from albumentations.pytorch import ToTensorV2

"""
extract path features
- WSI id1
    - patch1 feat.pkl --> dict({'val': [1280] , 'tr': })
    - patch2 feat.pkl
- WSI id2
    - patch1 feat.pkl
    - patch2 feat.pkl

"""


tr_trans = alb.Compose([
    alb.Resize(224, 224),
    alb.RandomRotate90(),
    alb.RandomBrightnessContrast(),
    alb.HueSaturationValue(),
    # alb.GaussNoise(),
    # alb.GaussianBlur(),
    alb.HorizontalFlip(),
    alb.VerticalFlip(),
    # alb.CLAHE(),
    alb.CoarseDropout(max_holes=4),
    alb.Normalize(),
    ToTensorV2(),
])

val_trans = alb.Compose([
    alb.Resize(224, 224),
    alb.Normalize(),
    ToTensorV2(),
])

class PatchDataset(data_utils.Dataset):
    def __init__(self, img_fp_list, trans):
        print("dataset for imagenet")
        self.img_fp_list = img_fp_list
        self.trans = trans

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, idx):
        try:
            img_fp = self.img_fp_list[idx]
            img = cv2.imread(img_fp)
            img = img[:, :, ::-1]
        except:
            # if img is None:
            img = np.zeros((224, 224, 3)).astype('uint8')

        pid = osp.basename(osp.dirname(img_fp))
        img_bname = osp.basename(img_fp).rsplit('.', 1)[0]
        aug_img = self.trans(image=img)['image']
        return pid, img_bname, aug_img


class PatchDatasetCLR(data_utils.Dataset):
    def __init__(self, img_fp_list):
        print("dataset for CLR")
        self.img_fp_list = img_fp_list

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, idx):
        try:
            img_fp = self.img_fp_list[idx]
            img_pil = Image.open(img_fp)
            img = cv2.resize(np.array(img_pil), (224, 224))
        except:
            # if img is None:
            img = np.zeros((224, 224, 3)).astype('uint8')
        pid = osp.basename(osp.dirname(img_fp))
        img_bname = osp.basename(img_fp).rsplit('.', 1)[0]
        aug_img = transforms.functional.to_tensor(img)
        return pid, img_bname, aug_img

#########################################################################

def save_feat_in_thread(batch_pid, batch_img_bname, batch_tr_feat, save_dir):
    for b_idx, (pid, img_bname, tr_feat) in enumerate(zip(batch_pid, batch_img_bname, batch_tr_feat)):
        feat_save_dir = osp.join(save_dir, pid)
        os.makedirs(feat_save_dir, exist_ok=True)
        feat_save_name = osp.join(feat_save_dir, f'{img_bname}.pkl')

        if osp.exists(feat_save_name):
            try:
                with open(feat_save_name, 'rb') as infile:
                    save_dict = pickle.load(infile)
            except:
                save_dict = {}
        else:
            save_dict = {}

        tr_aug_feat = np.stack([tr_feat])
        if 'tr' in save_dict.keys():
            save_dict['tr'] = np.concatenate([tr_aug_feat, save_dict['tr']])
        else:
            save_dict['tr'] = tr_aug_feat

        # print(f'Save to : {feat_save_name}')
        with open(feat_save_name, 'wb') as outfile:
            pickle.dump(save_dict, outfile)

def save_val_feat_in_thread(batch_pid, batch_img_bname, batch_val_feat, save_dir):
    for b_idx, (pid, img_bname, val_feat) in enumerate(zip(batch_pid, batch_img_bname, batch_val_feat)):
        feat_save_dir = osp.join(save_dir, pid)
        os.makedirs(feat_save_dir, exist_ok=True)
        feat_save_name = osp.join(feat_save_dir, f'{img_bname}.pkl')

        if osp.exists(feat_save_name):
            with open(feat_save_name, 'rb') as infile:
               save_dict = pickle.load(infile)
        else:
            save_dict = {}

        save_dict['val'] = val_feat

        with open(feat_save_name, 'wb') as outfile:
            pickle.dump(save_dict, outfile)

def pred_and_save_with_dataloader(model, img_fp_list, local_rank, save_dir, batch_size=1, times=1, dataset_class="imagenet"):
    print("start dataloader..")
    # model.cuda(local_rank)
    model.eval()

    # added by zhenyulin
    import multiprocessing
    num_processes = multiprocessing.cpu_count()
    num_processes = min(48, num_processes)


    executor = ThreadPoolExecutor(max_workers=num_processes)


    val_dl = torch.utils.data.DataLoader(
        PatchDataset(img_fp_list, trans=val_trans) if dataset_class=="imagenet" else PatchDatasetCLR(img_fp_list),
        batch_size=batch_size,
        num_workers=num_processes,
        shuffle=False,
        drop_last=False
    )

    for batch in progress.track(val_dl):
        batch_pid, batch_img_bname, batch_tr_img = batch
        batch_tr_img = batch_tr_img.cuda()

        with torch.no_grad():
            val_feat = model(batch_tr_img)
            val_feat = val_feat.cpu().numpy()
            executor.submit(save_val_feat_in_thread, batch_pid, batch_img_bname, val_feat, save_dir)
    print("task finished..")
#########################################################################
class EffNet(nn.Module):
    def __init__(self, efname='efficientnet-b0', model_path=""):
        super(EffNet, self).__init__()
        self.model = EfficientNet.from_name(efname)
        print(f'Load pretrain model from {model_path}')
        self.model.load_state_dict(torch.load(model_path))

    def forward(self, data):
        bs = data.shape[0]
        feat = self.model.extract_features(data)
        feat = nn.functional.adaptive_avg_pool2d(feat, output_size=(1))
        feat = feat.view(bs, -1)
        return feat

#########################################################################

class PretrainedNet(nn.Module):
    def __init__(self, model_name='se_resnet50', model_path="", num_classes=2):
        super(PretrainedNet, self).__init__()
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, num_classes)
        print(f'Load pretrain model from {model_path}')
        self.model.load_state_dict(torch.load(model_path))

    def forward(self, data):
        bs = data.shape[0]
        feat = self.model.features(data)
        feat = nn.functional.adaptive_avg_pool2d(feat, output_size=(1))
        feat = feat.view(bs, -1)
        return feat # 2048

#########################################################################

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        return h
class FeatureResNetSimCLR(nn.Module):
    def __init__(self, model_path="", base_model='se_resnet50', out_dim=256):
        super(FeatureResNetSimCLR, self).__init__()
        self.model = ResNetSimCLR(base_model=base_model, out_dim=out_dim)
        print(f'Load pretrain model from {model_path}')
        pretrained_dict = torch.load(model_path)
        try:
            self.model.load_state_dict(pretrained_dict)
        except:
            pretrained_dict_new = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
            self.model.load_state_dict(pretrained_dict_new)

    def forward(self, data):
        bs = data.shape[0]
        feat = self.model.features(data)
        feat = feat.view(bs, -1)
        return feat # 512
class FeatureResNetMoCo(nn.Module):
    def __init__(self, model_path="", base_model='se_resnet50'):
        super(FeatureResNetMoCo, self).__init__()
        self.model = models.__dict__[base_model]()
        print(f'Load pretrain model from {model_path}')
        state_dict = torch.load(model_path, map_location="cpu")
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove `module.`
            name = k.replace("module.", "")
            new_state_dict[name] = v
        state_dict = new_state_dict
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = self.model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        self.features = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, data):
        bs = data.shape[0]
        feat = self.features(data)
        feat = feat.view(bs, -1)
        return feat # 512

#########################################################################

def get_model(model_name="", model_path="", num_classes=2):
    if model_name == "efficientnet-b0":
        model = EffNet(efname='efficientnet-b0', model_path=model_path)
    elif model_name== 'MoCo.resnet18':
        model = FeatureResNetMoCo(model_path=model_path, base_model="resnet18")
    elif model_name== "SimCLR.resnet18":
        model = FeatureResNetSimCLR(model_path=model_path, base_model="resnet18", out_dim=256)
    elif model_name== "Patch.seresnet50":
        model = PretrainedNet(model_name="se_resnet50", model_path=model_path, num_classes=num_classes)
    else:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained="imagenet")
        print(f'Load pretrain model from {model_path}')
    return model

#########################################################################

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI patch features extraction")
    parser.add_argument("--cfg", default= None, metavar="FILE", help="path to config file", type=str,)
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    update_default_cfg(cfg)

    wsi_patch_info = cfg.PATCH.IDX2IMG
    model_name = cfg.PRETRAIN.MODEL_NAME
    num_classes = cfg.PRETRAIN.NUM_CLASSES
    model_path = cfg.PRETRAIN.MODEL_PATH
    times = cfg.PRETRAIN.TRAIN_FEA_TIMES
    save_dir = cfg.PRETRAIN.SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    # creat model
    model = get_model(model_name=model_name, model_path=model_path, num_classes=num_classes) # eff, se_resnet50
    model = nn.DataParallel(model).cuda()

    # load path of original images
    print("load img path..")
    img_fp_list = []
    with open(wsi_patch_info,"rb") as fp:
        dt = pickle.load(fp)
        for k, v in dt.items():
            img_fp_list.extend(v)
    print(f'Len of img {len(img_fp_list)}')

    img_fp_list = sorted(img_fp_list)

    # extract feature to *.pkl
    if model_name == 'MoCo.resnet18' or model_name == "SimCLR.resnet18":
        dataset_class = "clr"
    else:
        dataset_class = "imagenet"
    pred_and_save_with_dataloader(model, img_fp_list, local_rank=0, save_dir=save_dir, batch_size=1, times=times, dataset_class=dataset_class)