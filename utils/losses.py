# Thanks to rwightman's timm package
# github.com:rwightman/pytorch-image-models
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


# class LabelSmoothingCrossEntropy(nn.Module):
#     """
#     NLL loss with label smoothing.
#     """

#     def __init__(self, smoothing=0.1):
#         """
#         Constructor for the LabelSmoothing module.
#         :param smoothing: label smoothing factor
#         """
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         assert smoothing < 1.0
#         self.smoothing = smoothing
#         self.confidence = 1. - smoothing

#     def _compute_losses(self, x, target):
#         log_prob = F.log_softmax(x, dim=-1)
#         nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -log_prob.mean(dim=-1)
#         loss = self.confidence * nll_loss + self.smoothing * smooth_loss
#         return loss

#     def forward(self, x, target):
#         return self._compute_losses(x, target).mean()

class LabelSmoothingCrossEntropy(nn.Module):
    y_int = True  # y interpolation

    def __init__(self,
                 eps: float = 0.1,  # The weight for the interpolation formula
                 # Manual rescaling weight given to each class passed to `F.nll_loss`
                 weight: Tensor = None,
                 reduction: str = 'mean'  # PyTorch reduction to apply to the output
                 ):
        # store_attr()
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.weight = weight
        self.reduction = reduction

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        "Apply `F.log_softmax` on output then blend the loss/num_classes(`c`) with the `F.nll_loss`"
        c = output.size()[1]
        log_preds = F.log_softmax(output, dim=1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            # We divide by that size at the return line so sum and not mean
            loss = -log_preds.sum(dim=1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target.long(), weight=self.weight, reduction=self.reduction)

    def activation(self, out: Tensor) -> Tensor:
        "`F.log_softmax`'s fused activation function applied to model output"
        return F.softmax(out, dim=-1)

    def decodes(self, out: Tensor) -> Tensor:
        "Converts model output to target format"
        return out.argmax(dim=-1)
