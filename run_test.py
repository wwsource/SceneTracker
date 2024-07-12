# @File: run_test.py
# @Project: SceneTracker
# @Author : wangbo
# @Time : 2024.07.12

import argparse
import torch.nn.functional as F

from model.model_scenetracker import SceneTracker
from model.util import reduce_masked_mean, reduce_masked_median
from data.dataset import *

@torch.no_grad()
def validate_odyssey(model, split='test'):
    print('Start testing splatflow on LSFOdyssey...')

    val_set = WWOdyssey(seq_len=-1, track_point_num=-1, split=split)
    data_num = len(val_set)
    print(f'Dataset length {data_num}')

    metrics_all = {'acc3d_010': 0.0, 'acc3d_020': 0.0, 'acc3d_040': 0.0, 'acc3d_080': 0.0, 'acc3d_8': 0.0, 'epe3d': 0.0,
                   'survival_3d_5': 0.0, 'median_3d': 0.0, 'd_avg': 0.0, 'survival': 0.0, 'median_l2': 0.0}
    count_all = 0

    for val_id in range(data_num):

        dataset_outs = val_set[val_id]

        rgbs = dataset_outs['rgbs'].unsqueeze(0)  # shape: (B, T, 3, H, W)
        deps = dataset_outs['deps'].unsqueeze(0)  # shape: (B, T, H, W)
        trajs_uv = dataset_outs['trajs_uv'].unsqueeze(0)  # shape: (B, T, N, 2)
        trajs_z = dataset_outs['trajs_z'].unsqueeze(0)  # shape: (B, T, N, 1)
        visibs = dataset_outs['visibs'].unsqueeze(0)  # shape: (B, T, N)
        valids = dataset_outs['valids'].unsqueeze(0)  # shape: (B, T, N)
        query_points = dataset_outs['query_points'].unsqueeze(0)  # shape: (B, N, 4)
        intris = dataset_outs['intris']  # shape: (B, T, 3, 3)
        extris = dataset_outs['extris']  # shape: (B, T, 4, 4)

        step_data = [rgbs, deps, trajs_uv, trajs_z, visibs, valids, query_points, intris, extris]
        rgbs, deps, trajs_uv, trajs_z, visibs, valids, query_points, intris, extris = [x.cuda() for x in step_data]

        queries = query_points.clone().float()

        B, T, C, H, W = rgbs.shape
        B, N, D = queries.shape

        interp_shape = (384, 512)
        rgbs_raw = rgbs
        rgbs = rgbs.reshape(B * T, C, H, W)
        rgbs = F.interpolate(rgbs, interp_shape, mode="bilinear")
        rgbs = rgbs.reshape(B, T, 3, interp_shape[0], interp_shape[1])
        deps = deps.reshape(B * T, 1, H, W)
        deps = F.interpolate(deps, interp_shape, mode="bilinear")
        deps = deps.reshape(B, T, 1, interp_shape[0], interp_shape[1])

        queries[:, :, 1] *= interp_shape[1] / W
        queries[:, :, 2] *= interp_shape[0] / H

        trajs_uv_e, trajs_z_e, __, __ = model.infer(
            model,
            input_list=[rgbs, deps, queries],
            iters=4,
            is_train=False,
        )

        trajs_uv_e[:, :, :, 0] *= W / float(interp_shape[1])
        trajs_uv_e[:, :, :, 1] *= H / float(interp_shape[0])  # shape: (B, T, N, 2)

        trajs_xyz_e = torch.zeros((B, T, N, 3), device=trajs_uv.device)
        trajs_xyz_e[:, :, :, 0] = trajs_z_e[:, :, :, 0] * (trajs_uv_e[:, :, :, 0] - intris[:, :, 0, 2][..., None]) / \
                                  intris[:, :, 0, 0][..., None]
        trajs_xyz_e[:, :, :, 1] = trajs_z_e[:, :, :, 0] * (trajs_uv_e[:, :, :, 1] - intris[:, :, 1, 2][..., None]) / \
                                  intris[:, :, 1, 1][..., None]
        trajs_xyz_e[:, :, :, 2] = trajs_z_e[:, :, :, 0]

        trajs_xyz = torch.zeros((B, T, N, 3), device=trajs_uv.device)
        trajs_xyz[:, :, :, 0] = trajs_z[:, :, :, 0] * (trajs_uv[:, :, :, 0] - intris[:, :, 0, 2][..., None]) / \
                                intris[:, :, 0, 0][..., None]
        trajs_xyz[:, :, :, 1] = trajs_z[:, :, :, 0] * (trajs_uv[:, :, :, 1] - intris[:, :, 1, 2][..., None]) / \
                                intris[:, :, 1, 1][..., None]
        trajs_xyz[:, :, :, 2] = trajs_z[:, :, :, 0]


        res = torch.norm(trajs_xyz_e[:, 1:] - trajs_xyz[:, 1:], dim=-1)
        epe3d = torch.mean(res).item()
        acc3d_010 = reduce_masked_mean((res < 0.10).float(), valids[:, 1:]).item() * 100.0
        acc3d_020 = reduce_masked_mean((res < 0.20).float(), valids[:, 1:]).item() * 100.0
        acc3d_040 = reduce_masked_mean((res < 0.40).float(), valids[:, 1:]).item() * 100.0
        acc3d_080 = reduce_masked_mean((res < 0.80).float(), valids[:, 1:]).item() * 100.0
        acc3d_8 = (acc3d_080 + acc3d_010 + acc3d_020 + acc3d_040) / 4

        sur_thr = 0.50
        dists = torch.norm(trajs_xyz_e - trajs_xyz, dim=-1)  # B,S,N
        dist_ok = 1 - (dists > sur_thr).float() * valids.squeeze(-1)  # B,S,N
        survival_3d_5 = torch.cumprod(dist_ok, dim=1)  # B,S,N
        survival_3d_5 = torch.mean(survival_3d_5).item() * 100.0

        dists_ = dists.permute(0, 2, 1).reshape(B * N, T)
        valids_ = valids.permute(0, 2, 1).reshape(B * N, T)
        median_3d = reduce_masked_median(dists_, valids_, keep_batch=True)
        median_3d = median_3d.mean().item()

        sx_ = W / 256.0
        sy_ = H / 256.0
        sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
        sc_pt = torch.from_numpy(sc_py).float().cuda()

        thrs = [1, 2, 4, 8, 16]
        d_sum = 0.0
        for thr in thrs:
            # note we exclude timestep0 from this eval
            d_ = (torch.norm(trajs_uv_e[:, 1:] / sc_pt - trajs_uv[:, 1:] / sc_pt, dim=-1) < thr).float()  # B,S-1,N
            d_ = reduce_masked_mean(d_, valids[:, 1:]).item() * 100.0
            d_sum += d_
        d_avg = d_sum / len(thrs)

        sur_thr = 16
        dists = torch.norm(trajs_uv_e / sc_pt - trajs_uv / sc_pt, dim=-1)  # B,S,N
        dist_ok = 1 - (dists > sur_thr).float() * valids.squeeze(-1)  # B,S,N
        survival = torch.cumprod(dist_ok, dim=1)  # B,S,N
        survival = torch.mean(survival).item() * 100.0

        # get the median l2 error for each trajectory
        dists_ = dists.permute(0, 2, 1).reshape(B * N, T)
        valids_ = valids.permute(0, 2, 1).reshape(B * N, T)
        median_l2 = reduce_masked_median(dists_, valids_, keep_batch=True)
        median_l2 = median_l2.mean().item()

        metrics_tmp = {}
        metrics_tmp['acc3d_010'] = acc3d_010
        metrics_tmp['acc3d_020'] = acc3d_020
        metrics_tmp['acc3d_040'] = acc3d_040
        metrics_tmp['acc3d_080'] = acc3d_080
        metrics_tmp['acc3d_8'] = acc3d_8
        metrics_tmp['epe3d'] = epe3d
        metrics_tmp['survival_3d_5'] = survival_3d_5
        metrics_tmp['median_3d'] = median_3d
        metrics_tmp['d_avg'] = d_avg
        metrics_tmp['survival'] = survival
        metrics_tmp['median_l2'] = median_l2
        count_all += 1

        for key in metrics_all:
            number = metrics_tmp[key]
            metrics_all[key] += number

        if val_id % 5 == 0:
            print(f'{val_id}/{data_num}')

    for key in metrics_all:
        number = metrics_all[key] / count_all
        print("Odyssey (%s): %f" % (key, number))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='odyssey')
    parser.add_argument('--pre_name_path', default='exp/train_odyssey/model.pth')

    args = parser.parse_args()

    model = SceneTracker()
    pre_replace_list = [['module.', '']]
    checkpoint = torch.load(args.pre_name_path)
    for l in pre_replace_list:
        checkpoint = {k.replace(l[0], l[1]): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    print(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.eval().cuda()

    if args.dataset == 'odyssey':
        validate_odyssey(model)



