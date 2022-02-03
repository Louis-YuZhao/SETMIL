"""
    author
    Implemented from https://github.com/mathiaszinnen/focal_loss_torch


    Here is gamma  controls the shape of the curve.
    The higher the value of gamma, the lower the loss for well-classified examples,
        so we could turn the attention of the model more towards ‘hard-to-classify examples.
    Having higher gamma extends the range in which an example receives low loss.

    Another way, apart from Focal Loss, to deal with class imbalance is to introduce weights.
    Give high weights to the rare class and small weights to the dominating or common class.
    These weights are referred to as alpha.

"""

from torch import nn
import torch
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, reduction="mean"):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.reduction = reduction
        if isinstance(alpha, list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {} --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {}  --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            for n in range(1, num_classes):
                self.alpha[n] = 1-alpha

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        # gather : get prob from certain label-column
        preds_softmax = preds_softmax.gather(1, labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1,1))

        alpha = self.alpha.gather(0, labels.view(-1))


        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.reduction =="mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

_criterion = nn.BCEWithLogitsLoss()
class BCEWithLogitsLossWrapper(torch.nn.Module):
    def forward(self, preds, labels):
        return _criterion(preds, labels.float())
