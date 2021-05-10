'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2019-11-18 22:13:22
LastEditTime: 2020-12-21 12:00:10
@Description:
'''

import torch.nn as nn
import chamfer3D.dist_chamfer_3D


class ChamferDistLoss(nn.Module):
    """ Code from https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
    """
    def __init__(self):
        super().__init__()
        self.chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

    def forward(self, predict, target):   
        dist1, dist2, idx1, idx2 = self.chamLoss(predict[0][None], target[0][None])
        loss = dist1.mean() + dist2.mean()
        return loss
