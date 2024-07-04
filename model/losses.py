# @File: losses.py
# @Project: SceneTracker
# @Author : wangbo
# @Time : 2024.07.04

import torch
from .util import reduce_masked_mean

EPS = 1e-6

def sequence_loss(uv_preds, z_preds, uvz_gt, vis, valids, gamma=0.8, Z_WEIGHT=100):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0

    for j in range(len(uvz_gt)):

        B, S, N, D = uvz_gt[j].shape
        assert D == 3

        uv_gt, z_gt = torch.split(uvz_gt[j], [2, 1], dim=-1)  # shape: (B, S, N, 2) & (B, S, N, 1)

        B, S1, N = vis[j].shape
        B, S2, N = valids[j].shape
        assert S == S1
        assert S == S2
        n_predictions = len(uv_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            uv_pred = uv_preds[j][i]
            z_pred = z_preds[j][i]

            i_loss = (uv_pred - uv_gt).abs()  # B, S, N, 2
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N

            i_loss += Z_WEIGHT * (1.0 / z_pred - 1.0 / z_gt).abs().squeeze(-1)  # B, S, N, 1

            flow_loss += i_weight * reduce_masked_mean(i_loss, valids[j])
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss / float(N)
    return total_flow_loss