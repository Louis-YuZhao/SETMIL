import os
import os.path as osp
import json
import time
import copy
import pprint
import random
import zipfile
import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_t2t_irpe
import util.misc as utils
from util.loss import FocalLoss
from util.logger import getLog
from configs import get_cfg_defaults, update_default_cfg


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        if 'data_archive' in root or 'vis' in root or 'eps':
            # print(f'Ignore {root}')
            continue
        for file in files:
            write_fp = os.path.join(root, file)
            if 'core.' in write_fp:
                continue
            ziph.write(write_fp)

def main(cfg):

    time_now = datetime.datetime.now()

    unique_comment = f'{cfg.MODEL.MODEL_NAME}{time_now.month}{time_now.day}{time_now.hour}{time_now.minute}'
    cfg.TRAIN.OUTPUT_DIR = osp.join(cfg.TRAIN.OUTPUT_DIR, f'log_dataset_{cfg.DATASET.DATASET_SEED}', unique_comment)
    os.makedirs(cfg.TRAIN.OUTPUT_DIR, exist_ok=True)

    writer = SummaryWriter(log_dir=f'/*/log/tb/{unique_comment}', comment=f"{unique_comment}")
    logger = getLog(cfg.TRAIN.OUTPUT_DIR +"/log.txt", screen=True)

    # TODO
    utils.init_distributed_mode(cfg)
    logger.info("git:\n  {}\n".format(utils.get_sha()))

    logger.info(cfg)

    device = torch.device("cuda")

    # fix the seed for reproducibility
    seed = cfg.TRAIN.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model, criterion, postprocessors = build_model(args)
    model_name = cfg.MODEL.MODEL_NAME
    logger.info(f'Build Model: {model_name}')
    if model_name == 't2t_irpe':
        model, criterion = build_t2t_irpe(cfg)
        model.to(device)
    else:
        logger.info(f'Model name not found, {model_name}')
        raise ValueError(f'Model name not found')

    if cfg.TRAIN.LOSS_NAME == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2, num_classes=cfg.MODEL.NUM_CLASSES, reduction="mean")
    elif cfg.TRAIN.LOSS_NAME == "be":
        criterion = nn.CrossEntropyLoss()
    elif cfg.TRAIN.LOSS_NAME == "dsmil":
        pass
    elif cfg.TRAIN.LOSS_NAME == "cox":
        from util.coxloss import CoxPHLossWrapper
        criterion = CoxPHLossWrapper()

    if cfg.MULTI_LABEL:
        from util.loss import BCEWithLogitsLossWrapper
        criterion = BCEWithLogitsLossWrapper()


    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params:  {n_parameters}')

    dataset_train = build_dataset(image_set='train', args=cfg)
    dataset_val = build_dataset(image_set='val', args=cfg)
    dataset_test = build_dataset(image_set='test', args=cfg)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    collate_func = utils.collate_fn_multi_modal if cfg.DATASET.DATASET_NAME in ('vilt', 'vilt_surv', 'unit') else utils.collate_fn
    
    data_loader_train = DataLoader(dataset_train, shuffle=True,
                                   collate_fn=collate_func, num_workers=cfg.TRAIN.NUM_WORKERS,
                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                   pin_memory=True)

    data_loader_val = DataLoader(dataset_val, cfg.TRAIN.BATCH_SIZE, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_func, num_workers=cfg.TRAIN.NUM_WORKERS,
                                 pin_memory=True)

    data_loader_test = DataLoader(dataset_test, cfg.TRAIN.BATCH_SIZE, sampler=sampler_test,
                                 drop_last=False, collate_fn=collate_func, num_workers=cfg.TRAIN.NUM_WORKERS,
                                 pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, cfg.MODEL.LR_BACKBONE_NAME) and not match_name_keywords(n, cfg.MODEL.LR_LINEAR_PROJ_NAME) and p.requires_grad],
            "lr": cfg.TRAIN.LR,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.MODEL.LR_BACKBONE_NAME) and p.requires_grad],
            "lr": cfg.TRAIN.LR_BACKBONE,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.MODEL.LR_LINEAR_PROJ_NAME) and p.requires_grad],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.LR_LINEAR_PROJ_MULT,
        }
    ]
    if cfg.TRAIN.OPTIM_NAME=="sgd":
        optimizer = torch.optim.SGD(param_dicts, lr=cfg.TRAIN.LR, momentum=0.9,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIM_NAME=="adamw":
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model_without_ddp.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP)

    try:
        nni_id = os.environ['NNI_TRIAL_JOB_ID']
    except:
        nni_id = ''

    output_dir = Path(cfg.TRAIN.OUTPUT_DIR, nni_id)
    os.makedirs(output_dir, exist_ok=True)
    args_dict = vars(args)
    with open(osp.join(output_dir, 'params.txt'), 'wt') as outfile:
        pprint.pprint(args_dict, indent=4, stream=outfile)

    # backup file
    backup_fname = osp.join(cfg.TRAIN.OUTPUT_DIR, 'code_backup.zip')
    c_dir_path = os.path.dirname(os.path.realpath(__file__))
    logger.info(f'Backup file from {c_dir_path} to {backup_fname}')
    zipf = zipfile.ZipFile(backup_fname, 'w', zipfile.ZIP_DEFLATED)
    zipdir(c_dir_path, zipf)
    zipf.close()
    logger.info(f'Backup finish')

    # load if checkpoint
    if cfg.TRAIN.RESUME_PATH:
        logger.info(f'resume from {cfg.TRAIN.RESUME_PATH}')
        if cfg.TRAIN.RESUME_PATH.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.TRAIN.RESUME_PATH, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.TRAIN.RESUME_PATH, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            logger.info('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            logger.info('Unexpected Keys: {}'.format(unexpected_keys))
        if not cfg.TRAIN.EVAL and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            logger.info(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            cfg.override_resumed_lr_drop = True
            if cfg.override_resumed_lr_drop:
                logger.info('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = cfg.TRAIN.LR_DROP
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            cfg.TRAIN.START_EPOCH = checkpoint['epoch'] + 1

    # only eval
    if cfg.TRAIN.EVAL:
        test_stats, eval_result = evaluate(logger, model, criterion,
                                              data_loader_test, device, output_dir,
                                               cfg.distributed,
                                               display_header="Test",
                                           is_last_eval=True,
                                           save_path=cfg.TRAIN.OUTPUT_DIR,
                                           kappa_flag=cfg.TRAIN.KAPPA,
                                           )

        # save possibility to output_dir
        best_test_result = eval_result
        pred = best_test_result["pred"]
        for i in range(pred.shape[1]):
            best_test_result[f"pred_{i}"] = pred[:, i]
        del best_test_result["pred"]
        df_result = pd.DataFrame.from_dict(best_test_result)
        df_result.to_csv(output_dir / 'best_test_result.csv')

        if output_dir:
            utils.save_on_master(test_stats, output_dir / "eval.pth")
            utils.save_on_master(eval_result, output_dir / "eval_detail.pth")
        return

    # only training
    logger.info("Start training")
    start_time = time.time()
    bst_test_auc = -100
    bst_test_acc = -100
    bst_test_f1 = -100
    bst_test_rc = -100
    bst_test_pr = -100
    bst_val_auc = -100
    bst_epoch = 0
    best_model = None
    best_test_result = None
    max_patience = cfg.TRAIN.MAX_PATIENCE
    patience = 0

    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):

        if patience<max_patience:

            train_stats, train_result = train_one_epoch(logger,
                model, criterion, data_loader_train, optimizer, device, epoch, cfg.TRAIN.CLIP_MAX_NORM, is_distributed=cfg.distributed)
            lr_scheduler.step()


            eval_stats, eval_result = evaluate(logger,
                model, criterion, data_loader_val, device, output_dir, cfg.distributed,
                display_header="Valid",kappa_flag=cfg.TRAIN.KAPPA,
            )

            test_stats, test_result = evaluate(logger,
                model, criterion, data_loader_test, device, output_dir, cfg.distributed,
                display_header="Test",kappa_flag=cfg.TRAIN.KAPPA,
            )
            if cfg.TRAIN.KAPPA:
                val_auc = eval_stats['kappa']
                test_auc = test_stats['kappa']
            else:
                val_auc = eval_stats['auc']
                test_auc = test_stats['auc']
            test_acc = test_stats['acc']
            test_f1 = test_stats['f1']
            test_rc = test_stats['recall']
            test_pr = test_stats['precision']

            if bst_val_auc < val_auc:

                best_model = copy.deepcopy(model_without_ddp)
                bst_epoch = epoch
                bst_test_auc = test_auc
                bst_test_acc = test_acc
                bst_test_f1 = test_f1
                bst_test_rc = test_rc
                bst_test_pr = test_pr
                best_test_result = test_result

                pred = best_test_result["pred"]
                for i in range(pred.shape[1]):
                    best_test_result[f"pred_{i}"] = pred[:, i]
                del best_test_result["pred"]

                patience = 0
            else:
                patience += 1
            bst_val_auc = max(bst_val_auc, val_auc)

            bst_model_unique_comment = "checkpoint_test.pth"
            bst_model_checkpoint_path = output_dir / bst_model_unique_comment
            utils.save_on_master({
                'model': best_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': cfg,
                }, bst_model_checkpoint_path)

            bst_test_result_path = output_dir / 'best_test_result.csv'
            df_result = pd.DataFrame.from_dict(best_test_result)
            df_result.to_csv(bst_test_result_path)

            # log
            log_stats = {**{f'2_train_{k}': v for k, v in train_stats.items()},
                         **{f'3_eval_{k}': v for k, v in eval_stats.items()},
                         **{f'4_test_{k}': v for k, v in test_stats.items()},
                         '1_a_epoch': epoch,
                         '6_n_parameters': n_parameters,
                         '5_bst_epoch': bst_epoch,
                         }

            # logger.info(log_stats)
            # tensorboard logging
            writer.add_scalar('AUC/train', train_stats["auc"], global_step=epoch)
            writer.add_scalar('AUC/valid', eval_stats["auc"], global_step=epoch)
            writer.add_scalar('AUC/Test',  test_stats["auc"], global_step=epoch)
            writer.add_scalar('Accuracy/train', train_stats["acc"], global_step=epoch)
            writer.add_scalar('Accuracy/valid', eval_stats["acc"], global_step=epoch)
            writer.add_scalar('Accuracy/Test',  test_stats["acc"], global_step=epoch)
            writer.add_scalar('F1/train', train_stats["f1"], global_step=epoch)
            writer.add_scalar('F1/valid', eval_stats["f1"], global_step=epoch)
            writer.add_scalar('F1/Test',  test_stats["f1"], global_step=epoch)

            if output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats, sort_keys=True, indent=4) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    if cfg.TRAIN.KAPPA:
        logger.info(f'Best Val Cohen Kappa: {bst_val_auc:.4f} Test Cohen Kappa: {bst_test_auc:.4f} Accuracy: {bst_test_acc:.4f} at Epoch {bst_epoch}')
    else:
        logger.info(f'Best Val AUC: {bst_val_auc:.4f} Test Accuracy: {bst_test_acc:.4f} AUC: {bst_test_auc:.4f} Precision: {bst_test_pr:.4f}  Recall: {bst_test_rc:.4f}  F1: {bst_test_f1:.4f} at Epoch {bst_epoch}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="WSI training and evaluation script"
    )
    parser.add_argument(
        "--cfg",
        default="*.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    update_default_cfg(cfg)

    main(cfg)
