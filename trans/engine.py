from util.gpu_gather import GpuGather
from typing import Iterable
import torch
import util.misc as utils
import util.custom_metrics as custom_metrics
from sklearn import metrics
import numpy as np

from datasets.data_prefetcher import data_prefetcher


def train_one_epoch(logger, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, is_distributed=False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader) // 5
    print_freq = max(1, print_freq)

    gpu_gather = GpuGather(is_distributed=is_distributed)

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    # samples, targets = prefetcher.next()

    bag_samples, targets = prefetcher.next()

    IMG_id = []
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):

        # with torch.autograd.detect_anomaly():
        outputs = model(bag_samples)


        try:
            label = torch.tensor([x['label'].cpu().numpy() for x in targets])
        except:
            label = torch.tensor([x['label'] for x in targets])

        # logger.info(torch.unique(label))

        assert not np.any(np.isnan(label.cpu().numpy())), f"Label is nan"

        pid = torch.tensor([x['pid'] for x in targets])
        loss = criterion(outputs, label.cuda())

        # added by zhenyulin for dsmil
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        b = outputs.size(0)
        pred = torch.softmax(outputs.detach().cpu().view(b, -1), dim=1).numpy()
        gpu_gather.update(pred=pred)
        gpu_gather.update(label=label.cpu().numpy().reshape(-1))

        loss_dict = {'loss': loss}
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_value = loss_dict_reduced['loss'].item()


        optimizer.zero_grad()
        loss.backward()



        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()


        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(loss=loss_value)
        bag_samples, targets = prefetcher.next()

    gpu_gather.synchronize_between_processes()

    pred = gpu_gather.pred
    pred = np.concatenate(pred)
    num_of_class = pred.shape[1]

    label = gpu_gather.label
    label = np.concatenate(label)



    # multi-label
    if len(label) > len(pred):
        label = label.reshape(len(pred), -1)

        auc = []
        for i in range(num_of_class):
            pred_i = pred[:, i]
            # pred_i = torch.nn.sigmoid(pred_i)
            labek_i = label[:, i]
            if np.sum(labek_i) > 0:
                auc_i = metrics.roc_auc_score(labek_i, pred_i)
                auc.append(auc_i)


        logger.info(f'Train AUC: {auc}')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # logger.info("Averaged stats:", metric_logger)
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats['auc'] = np.mean(auc)
        stats['acc'] = 0
        stats['f1'] = 0
        stats['recall'] = 0
        stats['precision'] = 0


        eval_result = {}
        eval_result['label'] = label
        eval_result['pred'] = pred

        #modified by sunkai
        #eval_result['pid'] = pid
        eval_result["img_id"] = IMG_id

        return stats, eval_result
    else:
        # auc
        if num_of_class == 1:
            auc = custom_metrics.c_index(label, pred)
        elif num_of_class == 2:
            auc = metrics.roc_auc_score(label, pred[:, 1])
        else:
            auc = []
            for i in range(num_of_class):
                pred_i = pred[:,i]
                #pred_i = torch.nn.sigmoid(pred_i)
                labek_i = np.eye(num_of_class)[label][:, i]
                if np.sum(labek_i)>0:
                    auc_i = metrics.roc_auc_score(labek_i, pred_i)
                    auc.append(auc_i)
            auc = np.mean(auc)

    if num_of_class > 1:
        # cls report % confusion matrix
        cls_report = metrics.classification_report(label, np.argmax(pred, axis=1))
        cfm = metrics.confusion_matrix(label, np.argmax(pred, axis=1))
        logger.info(f"report: \n {cls_report}")
        logger.info(f"confusion matrix: \n {cfm}")

        pred_label = np.argmax(pred, axis=1)
        acc_score = metrics.accuracy_score(label, pred_label)
        if num_of_class == 2:
            f1_score = metrics.f1_score(label, pred_label)
            recall_score = metrics.recall_score(label, pred_label)
            precision_score = metrics.precision_score(label, pred_label)
        else:
            f1_score = metrics.f1_score(label, pred_label, average="weighted", zero_division=1)
            recall_score = metrics.recall_score(label, pred_label,average="weighted", zero_division=1)
            precision_score = metrics.precision_score(label, pred_label,average="weighted", zero_division=1)
        logger.info(f'AUC: {auc:.4f} Acc: {acc_score:.4f} F1: {f1_score:.4f} Recall: {recall_score:.4f} Precision: {precision_score:.4f}')

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # logger.info("Averaged stats:", metric_logger)
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats['auc'] = auc
        stats['acc'] = acc_score
        stats['f1'] = f1_score
        stats['recall'] = recall_score
        stats['precision'] = precision_score

        eval_result = {}
        eval_result['label'] = label
        eval_result['pred'] = pred
        eval_result["img_id"] =IMG_id

    else:
        logger.info(f'C-index: {auc:.4f}')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # logger.info("Averaged stats:", metric_logger)
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats['auc'] = auc
        stats['acc'] = 0
        stats['f1'] = 0
        stats['recall'] = 0
        stats['precision'] = 0

        eval_result = {}
        eval_result['label'] = label
        eval_result['pred'] = pred
        eval_result['pid'] = pid
    return stats, eval_result


@torch.no_grad()
def model_vis(model, criterion, data_loader, device, output_dir, is_distributed, display_header="Model Visual:"):
    model.eval()
    criterion.eval()
    header = f'{display_header}:'
    metric_logger = utils.MetricLogger(delimiter="  ")

    print_freq = len(data_loader) // 4
    print_freq = max(1, print_freq)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        label = torch.tensor([x['label'] for x in targets])
        pid = torch.tensor([x['pid'] for x in targets])

        output = model(samples)

        ref_offset = model.transformer.encoder[0].MSDeformAttn.sampling_offsets
        att_point  = model.transformer.encoder[0].MSDeformAttn.attention_weights

        print(ref_offset)
        print('==' * 5)
        print(att_point)


@torch.no_grad()
def evaluate(logger, model, criterion, data_loader, device, output_dir, is_distributed, display_header="Valid", is_last_eval=False, save_path='', kappa_flag=False):
    model.eval()
    # model.train()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'{display_header}:'

    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    gpu_gather = GpuGather(is_distributed=is_distributed)
    print_freq = len(data_loader) // 4
    print_freq = max(1, print_freq)

    IMG_id = []

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        #modified by sunkai
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            label = torch.tensor([x['label'].cpu().numpy() for x in targets])
        except:
            label = torch.tensor([x['label'] for x in targets])

        pid = torch.tensor([x['pid'] for x in targets])
        IMG_id+=pid
        #modified bu sunkai
        #img_id = [x['img_id'] for x in targets]
        #IMG_id+=img_id

        try:
            outputs = model(samples)
        except Exception as e:
            logger.info(samples['target'])
            raise e
        # loss = criterion(outputs, targets)

        loss = criterion(outputs, label.cuda())

        loss_dict = {'loss': loss}
        # weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        metric_logger.update(loss=loss_dict_reduced['loss'].item())
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])


        # added by zhenyulin for dsmil
        if isinstance(outputs, tuple):
            outputs = outputs[0]


        b = outputs.size(0)
        #gpu_gather.update(pred=torch.softmax(outputs.detach().cpu().view(b, -1), dim=1).numpy())
        gpu_gather.update(pred=outputs.detach().cpu().view(b, -1).numpy())
        gpu_gather.update(label=label.cpu().numpy().reshape(-1))

        # modified by sunkai
        #gpu_gather.update(pid=pid.cpu().numpy().tolist())
        #gpu_gather.update(img_id=img_id.cpu().numpy().tolist())



    # gather the stats from all processes
    panoptic_res = None
    gpu_gather.synchronize_between_processes()

    pred = gpu_gather.pred
    pred = np.concatenate(pred)
    num_of_class = pred.shape[1]

    label = gpu_gather.label
    label = np.concatenate(label)

    # modified by sunkai
    # pid = gpu_gather.pid
    # pid = np.array(pid)
    # logger.info(f'number of example: {pid.shape}')




    # logger.info(f'{pred.shape}, {label.shape}, ')
    # keep only unique (and in sorted order) images
    # merged_img_ids, idx = np.unique(pid, return_index=True)
    # label = label[idx]
    # pred = pred[idx]


    # multi-label
    if len(label) > len(pred):
        label = label.reshape(len(pred), -1)

        auc = []
        for i in range(num_of_class):
            pred_i = pred[:, i]
            # pred_i = torch.nn.sigmoid(pred_i)
            labek_i = label[:, i]
            if np.sum(labek_i) > 0:
                auc_i = metrics.roc_auc_score(labek_i, pred_i)
                auc.append(auc_i)


        logger.info(f'{display_header} AUC: {auc}')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # logger.info("Averaged stats:", metric_logger)
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats['auc'] = np.mean(auc)
        stats['acc'] = 0
        stats['f1'] = 0
        stats['recall'] = 0
        stats['precision'] = 0

        eval_result = {}
        eval_result['label'] = label
        eval_result['pred'] = pred
        #eval_result['pid'] = pid

        #modified by sunkai
        eval_result['imd_id'] = IMG_id

        return stats, eval_result
    else:
        try:
            if num_of_class == 1:
                auc = custom_metrics.c_index(label, pred)
            elif num_of_class == 2:
                auc = metrics.roc_auc_score(label, pred[:, 1])
            else:
                auc = []
                for i in range(num_of_class):
                    pred_i = pred[:,i]
                    #pred_i = torch.nn.sigmoid(pred_i)
                    labek_i = np.eye(num_of_class)[label][:, i]
                    if np.sum(labek_i)>0:
                        auc_i = metrics.roc_auc_score(labek_i, pred_i)
                        auc.append(auc_i)
                auc = np.mean(auc)
        except Exception as e:
            print(e)
            auc = 0

    if num_of_class > 1:
        cls_report = metrics.classification_report(label, np.argmax(pred, axis=1))
        cfm = metrics.confusion_matrix(label, np.argmax(pred, axis=1))

        logger.info(f"report: \n {cls_report}")
        logger.info(f"confusion matrix: \n {cfm}")

        # if is_last_eval:
        #     import pandas as pd
        #     data = np.hstack([label.reshape(-1, 1), pred.reshape(-1, 2)])
        #     df = pd.DataFrame(data, columns=['target', 'pred_0', 'pred_1'])
        #     save_fp = os.path.join(save_path, 'pred.csv')
        #     logger.info(f'Save result to {save_fp}')
        #     df.to_csv(save_fp, index=False, encoding='utf_8_sig')

        pred_label = np.argmax(pred, axis=1)
        acc_score = metrics.accuracy_score(label, pred_label)
        if num_of_class == 2:
            # precision, recall, thresholds = metrics.precision_recall_curve(label, pred[:, 1])
            # f1_scores = 2 * recall * precision / (recall + precision)
            # # if is_last_eval:
            # #     for f, r, p in zip(f1_scores, recall, precision):
            # #         logger.info(f'{f:.4f} {p: .4f} {r:.4f}')
            # f1_scores[np.isnan(f1_scores)] = 0
            # bst_idx = np.argmax(f1_scores)
            # logger.info(
            #     f'{display_header} Best F1: {f1_scores[bst_idx]:.4f} Precision: {precision[bst_idx]:.4f} Recall: {recall[bst_idx]:.4f}')
            f1_score = metrics.f1_score(label, pred_label)
            recall_score = metrics.recall_score(label, pred_label)
            precision_score = metrics.precision_score(label, pred_label)
        else:
            f1_score = metrics.f1_score(label, pred_label, average="weighted", zero_division=1)
            recall_score = metrics.recall_score(label, pred_label, average="weighted", zero_division=1)
            precision_score = metrics.precision_score(label, pred_label, average="weighted", zero_division=1)

        logger.info(f'{display_header} AUC: {auc:.4f} Acc: {acc_score:.4f} F1: {f1_score:.4f} Recall: {recall_score:.4f} Precision: {precision_score:.4f}')

        if kappa_flag:
            logger.info(f"ground truth: {label}, prediction: {pred_label}")
            kappa_res = custom_metrics.quadratic_kappa(label, pred_label, N=num_of_class)
            logger.info(f"quadratic kappa: \n {kappa_res}")
        # for cls_idx in range(num_of_class):
        #     current_prob = pred[:, cls_idx]
        #     current_label = (label == cls_idx)
        #     current_auc = metrics.roc_auc_score(current_label, current_prob)
        #     logger.info(f'OVR AUC of Class {cls_idx} {current_auc:.5f}')

        metric_logger.synchronize_between_processes()
        #logger.info("Averaged stats:", metric_logger)
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats['auc'] = auc
        stats['acc'] = acc_score
        stats['f1'] = f1_score
        stats['recall'] = recall_score
        stats['precision'] = precision_score
        if kappa_flag:
            stats['kappa'] = kappa_res

        eval_result = {}
        eval_result['label'] = label
        eval_result['pred'] = pred

        # modified by sunkai
        #eval_result['pid'] = pid
        eval_result['img_id'] = IMG_id

    else:
        logger.info(f'{display_header} C-index: {auc:.4f}')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # logger.info("Averaged stats:", metric_logger)
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats['auc'] = auc
        stats['acc'] = 0
        stats['f1'] = 0
        stats['recall'] = 0
        stats['precision'] = 0

        eval_result = {}
        eval_result['label'] = label
        eval_result['pred'] = pred

        #modified by sunkai
        #eval_result['pid'] = pid
        eval_result['imd_id'] = img_id
    return stats, eval_result
