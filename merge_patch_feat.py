import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import os.path as osp
import pickle
from glob import glob
from multiprocessing import Pool

from tqdm import tqdm
import rich
from rich import print
from rich.progress import track

"""
extract_patch_with_tta.py 将同一个WSI内offline提取的特征合并成单个文件
- WSI id1.pkl --> list([ dict('feat_name': 'RS1800102FFP-tile-r8705-c10752-512x512', 'val': [1280 origin] , 'tr': [ [aug1], []]), dict() ])
- WSI id2.pkl
"""

# select_patch_name = set()
#
# select_patch_dir = luad_select_patch_dir
#
# file_name = glob(osp.join(select_patch_dir, '*.txt'))
# for fp in file_name:
#     with open(fp, 'rt') as infile:
#         d = infile.readlines()
#         d = [x.strip() for x in d]
#         for x in d:
#             select_patch_name.add(osp.basename(x).rsplit('.', 1)[0].lower())
#




def merge_wsi_feat(wsi_feat_dir) -> None:
    """
    合并文件子进程, 每个进程处理一个WSI内的所有文件
    Args:
        wsi_feat_dir: WSI内Patch特征所在文件夹

    Returns:

    """

    # 查找WSI id 内所有Patch 特征

    files = glob(osp.join(wsi_feat_dir, '*.pkl'))

    # files_filter = [x for x in files if osp.basename(x).rsplit('.', 1)[0].lower() in select_patch_name]
    # if len(files) != len(files_filter):
    #     print(f'Filtered {len(files)} => {len(files_filter)}')
    # files = files_filter
    # print(f'Search in {wsi_feat_dir} && found {len(files)}')

    save_obj = []
    for fp in files:
        # 可能同时在运行extract文件, 导致个别文件的pickle序列不完整
        try:
            with open(fp, 'rb') as infile:
                obj = pickle.load(infile)

            # 添加patch的文件名，后续可视化时有用到
            obj['feat_name'] = osp.basename(fp).rsplit('.', 1)[0]
            # obj['val'] = obj['tr']
            save_obj.append(obj)
        except Exception as e:
            print(f'Error in {fp} as {e}')
            continue

    bname = osp.basename(wsi_feat_dir).lower()  # wsi id
    save_fp = osp.join(merge_feat_save_dir, f'{bname}.pkl')
    with open(save_fp, 'wb') as outfile:
        pickle.dump(save_obj, outfile)

from configs import get_cfg_defaults, update_default_cfg
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="WSI patch features extraction"
    )
    parser.add_argument(
        "--cfg",
        default="/aaa/louisyuzhao/guy1/kaikasun/goWSI/trans/configs/configs_kaikasun/huyanyuan_outside_patch_dtmil_KRAS_firstexternal_outside_test.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    update_default_cfg(cfg)

    # 上一步提取的特征保存位置
    feat_save_dirx20 = cfg.PRETRAIN.SAVE_DIR
    # 输出路径
    merge_feat_save_dirx20 = cfg.PRETRAIN.SAVE_DIR_COMBINED



    for feat_save_dir, merge_feat_save_dir in zip(
        [feat_save_dirx20, ],
        [merge_feat_save_dirx20]
    ):
        print(f'Save to {merge_feat_save_dir}')
        os.makedirs(merge_feat_save_dir, exist_ok=True)
        wsi_dirs = glob(osp.join(feat_save_dir, '*'))

        with Pool(80) as p:
            for _ in track(p.imap_unordered(merge_wsi_feat, wsi_dirs), total=len(wsi_dirs)):
                pass
