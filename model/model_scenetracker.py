# @File: model_scenetracker.py
# @Project: SceneTracker
# @Author : wangbo
# @Time : 2024.07.04

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .block import BasicEncoder, CorrBlock, UpdateFormer
from .embeddings import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, get_2d_embedding
from .losses import sequence_loss
from .util import bilinear_sample2d, smart_cat
autocast = torch.cuda.amp.autocast
enable_autocast = False
import torch.distributed as dist

def sample_pos_embed(grid_size, embed_dim, coords):
    pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size)  # shape: (h*w, embed_dim)
    pos_embed = (
        torch.from_numpy(pos_embed)
        .reshape(grid_size[0], grid_size[1], embed_dim)
        .float()
        .unsqueeze(0)
        .to(coords.device)
    )  # shape: (1, h, w, embed_dim)
    # coords shape: (B, S, N_before_window_end, 2)
    sampled_pos_embed = bilinear_sample2d(
        pos_embed.permute(0, 3, 1, 2), coords[:, 0, :, 0], coords[:, 0, :, 1]
    )  # shape: (B, embed_dim, N_before_window_end)
    return sampled_pos_embed

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class SceneTracker(nn.Module):
    def __init__(self, config=None):
        super(SceneTracker, self).__init__()
        self.config = config

        space_depth = 6
        time_depth = 6
        hidden_size = 384
        num_heads = 8
        add_space_attn = True

        self.stride = stride = 8
        self.S = 16

        self.latent_dim = latent_dim = 128
        self.corr_levels = 4
        self.corr_radius = 3


        self.fnet = nn.Sequential(
            BasicEncoder(output_dim=self.latent_dim, norm_fn="instance", dropout=0, stride=stride)
        )
        self.additional_dim = 4

        self.updateformer = nn.Sequential(
            UpdateFormer(
                space_depth=space_depth,
                time_depth=time_depth,
                input_dim=456 + self.additional_dim,
                hidden_size=hidden_size,
                num_heads=num_heads,
                output_dim=latent_dim + 2 + 1,
                mlp_ratio=4.0,
                add_space_attn=add_space_attn,
            ),
        )

        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )

        self.vis_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

        self.enable_autocast = enable_autocast

    def get_dcorrs(self, finvs, coords, invs_pred):

        # finvs shape: (B, S, 1, h, w)
        # coords shape: (B, S, N, 2)
        # invs_pred shape: (B, S, N, 1)
        B, S, _, H, W = finvs.shape
        N = coords.shape[2]
        coords = coords.view(B * S, N, 2).unsqueeze(2)  # shape: (B*S, N, 1, 2)
        finvs = finvs.view(B * S, 1, H, W)  # shape: (B*S, 1, H, W)

        invs_proj, _ = bilinear_sampler(finvs, coords, mask=True)
        invs_proj = invs_proj.view(B, S, N, 1)
        dcorrs = invs_proj - invs_pred  # shape: (B, S, N, corr_dim=1)

        return dcorrs  # shape: (B, S, N, corr_dim)

    def forward_iteration(self, fmaps, fdeps, coords_init, deps_pred_init, feat_init=None, vis_init=None,
                          track_mask=None, iters=4):

        B, S_init, N, D = coords_init.shape  # shape: (B, S, N_before_window_end, 2)
        assert D == 2
        assert B == 1

        B, S, __, H8, W8 = fmaps.shape  # shape: (B, S, C, h, w)
        # fdeps shape: (B, S, 1, h, w)

        device = fmaps.device

        if S_init < S:
            coords = torch.cat(
                [coords_init, coords_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
            deps_pred = torch.cat(
                [deps_pred_init, deps_pred_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
            vis_init = torch.cat(
                [vis_init, vis_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
        else:
            coords = coords_init.clone()  # shape: (B, S, N_before_window_end, 2)
            deps_pred = deps_pred_init.clone()  # shape: (B, S, N_before_window_end, 1)

        fcorr_fn = CorrBlock(
            fmaps, num_levels=self.corr_levels, radius=self.corr_radius
        )
        fdeps[fdeps < 0.01] = 85.0
        finvs = 1.0 / fdeps  # shape: (B, S, 1, h, w)

        ffeats = feat_init.clone()  # shape: (B, S, N_before_window_end, C)

        times_ = torch.linspace(0, S - 1, S).reshape(1, S, 1)  # shape: (1, S, 1), [0, 1, ..., S-1]

        pos_embed = sample_pos_embed(
            grid_size=(H8, W8),
            embed_dim=456 + self.additional_dim,
            coords=coords,  # shape: (B, S, N_before_window_end, 2)
        )  # shape: (B, embed_dim, N_before_window_end)
        pos_embed = rearrange(pos_embed, "b e n -> (b n) e").unsqueeze(
            1)  # shape: (B*N_before_window_end, 1, embed_dim)
        times_embed = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(456 + self.additional_dim, times_[0]))[None]
                .repeat(B, 1, 1)
                .float()
                .to(device)
        )  # shape: (B, S, embed_dim)
        coord_predictions = []
        dep_predictions = []

        for __ in range(iters):
            coords = coords.detach()  # shape: (B, S, N_before_window_end, 2)
            deps_pred = deps_pred.detach()  # shape: (B, S, N_before_window_end, 1)
            deps_pred[deps_pred < 0.01] = 0.01

            fcorr_fn.corr(ffeats)

            fcorrs = fcorr_fn.sample(coords)  # shape: (B, S, N_before_window_end, corr_dim)
            LRR = fcorrs.shape[3]

            dcorrs = self.get_dcorrs(finvs, coords, 1.0 / deps_pred)  # shape: (B, S, N_before_window_end, corr_dim)
            dcorrs_dim = dcorrs.shape[3]

            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)  # shape: (B*N_before_window_end, S, fcorrs_dim)
            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S,
                                                                           2)  # shape: (B*N_before_window_end, S, 2)

            dcorrs_ = dcorrs.permute(0, 2, 1, 3).reshape(B * N, S,
                                                         dcorrs_dim)  # shape: (B*N_before_window_end, S, dcorrs_dim)
            dep_flows_ = (deps_pred - deps_pred[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S,
                                                                                     1)  # shape: (B*N_before_window_end, S, 1)

            flows_cat = get_2d_embedding(flows_, 64, cat_coords=True)  # shape: (B*N_before_window_end, S, 2*pe_dim+2)
            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N, S,
                                                         self.latent_dim)  # shape: (B*N_before_window_end, S, C)

            if track_mask.shape[1] < vis_init.shape[1]:
                track_mask = torch.cat(
                    [
                        track_mask,
                        torch.zeros_like(track_mask[:, 0]).repeat(
                            1, vis_init.shape[1] - track_mask.shape[1], 1, 1
                        ),
                    ],
                    dim=1,
                )
            concat = (
                torch.cat([track_mask, vis_init], dim=2)
                    .permute(0, 2, 1, 3)
                    .reshape(B * N, S, 2)
            )

            transformer_input = torch.cat([flows_cat, fcorrs_, dep_flows_, dep_flows_, dcorrs_, dcorrs_, ffeats_, concat],
                                          dim=2)  # shape: (B*N_before_window_end, S, inp_dim)
            x = transformer_input + pos_embed + times_embed  # shape: (B*N_before_window_end, S, inp_dim)

            x = rearrange(x, "(b n) t d -> b n t d", b=B)  # shape: (B, N_before_window_end, S, inp_dim)

            with autocast(enabled=self.enable_autocast):
                delta = self.updateformer(x)  # shape: (B, N_before_window_end, S, C+2)

                delta = rearrange(delta, " b n t d -> (b n) t d")  # shape: (B*N_before_window_end, S, C+2)

                delta_coords_ = delta[:, :, :2]  # shape: (B*N_before_window_end, S, 2)
                delta_deps_pred_ = delta[:, :, 2:3]  # shape: (B*N_before_window_end, S, 1)
                delta_feats_ = delta[:, :, 3:]  # shape: (B*N_before_window_end, S, C)

                delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)  # shape: (B*N_before_window_end*S, C)
                ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N * S,
                                                             self.latent_dim)  # shape: (B*N_before_window_end*S, C)

                ffeats_ = self.ffeat_updater(
                    self.norm(delta_feats_)) + ffeats_  # shape: (B*N_before_window_end*S, C), feature_init is updated

                ffeats = ffeats_.reshape(B, N, S, self.latent_dim).permute(
                    0, 2, 1, 3
                )  # shape: (B, S, N_before_window_end, C)

                coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1,
                                                                            3)  # shape: (B, S, N_before_window_end, 2)
                coord_predictions.append(coords * self.stride)

                deps_pred = deps_pred + delta_deps_pred_.reshape(B, N, S, 1).permute(0, 2, 1,
                                                                                     3)  # shape: (B, S, N_before_window_end, 1)
                deps_pred[deps_pred < 0.01] = 0.01
                dep_predictions.append(deps_pred)

        with autocast(enabled=self.enable_autocast):
            vis_e = self.vis_predictor(ffeats.reshape(B * S * N, self.latent_dim)).reshape(
                B, S, N
            )  # shape: (B, S, N_before_window_end)
        vis_e = vis_e.float()
        return coord_predictions, dep_predictions, vis_e, feat_init

    def forward(self, rgbs, deps, queries, iters=4, feat_init=None, is_train=False):
        B, T, C, H, W = rgbs.shape  # shape: (B, T, 3, H, W)
        B, N, __ = queries.shape  # shape: (B, N, 3)

        device = queries.device
        assert B == 1
        # INIT for the first sequence
        # We want to sort points by the first frame they are visible to add them to the tensor of tracked points consequtively
        first_positive_inds = queries[:, :, 0].long()  # shape: (B, N)

        __, sort_inds = torch.sort(first_positive_inds[0], dim=0, descending=False)  # shape: (N)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        first_positive_sorted_inds = first_positive_inds[0][sort_inds]

        assert torch.allclose(
            first_positive_inds[0], first_positive_inds[0][sort_inds][inv_sort_inds]
        )

        coords_init = queries[:, :, 1:3].reshape(B, 1, N, 2).repeat(
            1, self.S, 1, 1
        ) / float(self.stride)  # shape: (B, S, N, 2)
        deps_pred_init = queries[:, :, 3:4].reshape(B, 1, N, 1).repeat(
            1, self.S, 1, 1
        )  # shape: (B, S, N, 1)

        rgbs = 2 * (rgbs / 255.0) - 1.0

        traj_uv_e = torch.zeros((B, T, N, 2), device=device)  # trajs_g shape: (B, T, N, 2)
        traj_z_e = torch.zeros((B, T, N, 1), device=device)  # trajs_g shape: (B, T, N, 1)

        ind_array = torch.arange(T, device=device)
        ind_array = ind_array[None, :, None].repeat(B, 1, N)  # shape: (B, T, N)

        track_mask = (ind_array >= first_positive_inds[:, None, :]).unsqueeze(-1)  # shape: (B, T, N, 1)
        # these are logits, so we initialize visibility with something that would give a value close to 1 after softmax
        vis_init = torch.ones((B, self.S, N, 1), device=device).float() * 10  # shape: (B, S, N, 1)

        ind = 0

        track_mask_ = track_mask[:, :, sort_inds].clone()
        coords_init_ = coords_init[:, :, sort_inds].clone()
        deps_pred_init_ = deps_pred_init[:, :, sort_inds].clone()
        vis_init_ = vis_init[:, :, sort_inds].clone()

        prev_wind_idx = 0
        fmaps_ = None
        coord_predictions = []
        dep_predictions = []
        wind_inds = []
        while ind < T - self.S // 2:
            rgbs_seq = rgbs[:, ind: ind + self.S]  # shape: (B, S, 3, H, W)
            deps_seq = deps[:, ind: ind + self.S]  # shape: (B, S, 1, H, W)
            if rgbs_seq.device == torch.device('cpu'):
                rgbs_seq = rgbs_seq.cuda()
                deps_seq = deps_seq.cuda()

            S = S_local = rgbs_seq.shape[1]
            if S < self.S:
                rgbs_seq = torch.cat(
                    [rgbs_seq, rgbs_seq[:, -1, None].repeat(1, self.S - S, 1, 1, 1)],
                    dim=1,
                )
                deps_seq = torch.cat(
                    [deps_seq, deps_seq[:, -1, None].repeat(1, self.S - S, 1, 1, 1)],
                    dim=1,
                )
                S = rgbs_seq.shape[1]
            rgbs_ = rgbs_seq.reshape(B * S, C, H, W)  # shape: (B*S, 3, H, W)
            deps_ = deps_seq.reshape(B * S, 1, H, W)  # shape: (B*S, 1, H, W)

            if fmaps_ is None:
                with autocast(enabled=self.enable_autocast):
                    fmaps_ = self.fnet(rgbs_)  # shape: (B*S, C, h, w)
                fdeps_ = deps_[:, :, ::8, ::8]  # shape: (B*S, 1, h, w)
            else:
                with autocast(enabled=self.enable_autocast):
                    fmaps_ = torch.cat(  # something wrong with batch_size > 1
                        [fmaps_[self.S // 2:], self.fnet(rgbs_[self.S // 2:])], dim=0
                    )
                fdeps_ = torch.cat(  # something wrong with batch_size > 1
                    [fdeps_[self.S // 2:], deps_[self.S // 2:][:, :, ::8, ::8]], dim=0
                )
            fmaps_ = fmaps_.float()
            fmaps = fmaps_.reshape(
                B, S, self.latent_dim, H // self.stride, W // self.stride
            )  # shape: (B, S, C, h, w)
            fdeps = fdeps_.reshape(
                B, S, 1, H // self.stride, W // self.stride
            )  # shape: (B, S, 1, h, w)

            curr_wind_points = torch.nonzero(
                first_positive_sorted_inds < ind + self.S)  # shape: (n, 1), n is the available point number.
            if curr_wind_points.shape[0] == 0:
                ind = ind + self.S // 2
                continue
            wind_idx = curr_wind_points[-1] + 1

            if wind_idx - prev_wind_idx > 0:
                fmaps_sample = fmaps[
                               :, first_positive_sorted_inds[prev_wind_idx:wind_idx] - ind
                               ]  # shape: (B, n, C, h, w)

                feat_init_ = bilinear_sample2d(
                    fmaps_sample,
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 0],
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 1],
                ).permute(0, 2, 1)  # shape: (B, n, C)

                feat_init_ = feat_init_.unsqueeze(1).repeat(1, self.S, 1, 1)  # shape: (B, S, n, C)
                feat_init = smart_cat(feat_init, feat_init_, dim=2)  # shape: (B, S, n_last+n, C)

            if prev_wind_idx > 0:
                new_coords = coords[-1][:, self.S // 2:] / float(self.stride)

                coords_init_[:, : self.S // 2, :prev_wind_idx] = new_coords
                coords_init_[:, self.S // 2:, :prev_wind_idx] = new_coords[
                                                                :, -1
                                                                ].repeat(1, self.S // 2, 1, 1)

                new_deps_pred = deps_pred[-1][:, self.S // 2:]
                deps_pred_init_[:, : self.S // 2, :prev_wind_idx] = new_deps_pred
                deps_pred_init_[:, self.S // 2:, :prev_wind_idx] = new_deps_pred[
                                                                   :, -1
                                                                   ].repeat(1, self.S // 2, 1, 1)

                new_vis = vis[:, self.S // 2:].unsqueeze(-1)
                vis_init_[:, : self.S // 2, :prev_wind_idx] = new_vis
                vis_init_[:, self.S // 2:, :prev_wind_idx] = new_vis[:, -1].repeat(
                    1, self.S // 2, 1, 1
                )

            coords, deps_pred, vis, __ = self.forward_iteration(
                fmaps=fmaps,  # shape: (B, S, C, h, w)
                fdeps=fdeps,  # shape: (B, S, 1, h, w)
                coords_init=coords_init_[:, :, :wind_idx],  # shape: (B, S, N_before_window_end, 2)
                deps_pred_init=deps_pred_init_[:, :, :wind_idx],  # shape: (B, S, N_before_window_end, 1)
                feat_init=feat_init[:, :, :wind_idx],  # shape: (B, S, N_before_window_end, C)
                vis_init=vis_init_[:, :, :wind_idx],  # shape: (B, S, N_before_window_end, 1)
                track_mask=track_mask_[:, ind: ind + self.S, :wind_idx],  # shape: (B, S, N_before_window_end, 1)
                iters=iters,
            )
            # coords: list, each element shape: (B, S, N_before_window_end, 2)
            if is_train:
                coord_predictions.append([coord[:, :S_local] for coord in coords])
                dep_predictions.append([dep_pred[:, :S_local] for dep_pred in deps_pred])
                wind_inds.append(wind_idx)

            traj_uv_e[:, ind: ind + self.S, :wind_idx] = coords[-1][:, :S_local]  # shape: (B, T, N, 2)
            traj_z_e[:, ind: ind + self.S, :wind_idx] = deps_pred[-1][:, :S_local]  # shape: (B, T, N, 1)

            track_mask_[:, : ind + self.S, :wind_idx] = 0.0
            ind = ind + self.S // 2

            prev_wind_idx = wind_idx

        traj_uv_e = traj_uv_e[:, :, inv_sort_inds]  # shape: (B, T, N, 2)
        traj_z_e = traj_z_e[:, :, inv_sort_inds]  # shape: (B, T, N, 2)

        train_data = (
            (coord_predictions, dep_predictions, wind_inds, sort_inds)
            if is_train
            else None
        )
        return traj_uv_e, traj_z_e, feat_init, train_data

    def Loss(self, train_data, gt_list):

        coord_predictions, dep_predictions, wind_inds, sort_inds = train_data
        trajs_g, vis_g, valids = gt_list

        trajs_g = trajs_g[:, :, sort_inds]  #  shape: (B, T, N, 3)
        vis_g = vis_g[:, :, sort_inds]  #  shape: (B, T, N)
        valids = valids[:, :, sort_inds]  #  shape: (B, T, N)

        vis_gts = []
        traj_gts = []
        valids_gts = []

        for i, wind_idx in enumerate(wind_inds):
            ind = i * (self.S // 2)

            vis_gts.append(vis_g[:, ind: ind + self.S, :wind_idx])
            traj_gts.append(trajs_g[:, ind: ind + self.S, :wind_idx])
            valids_gts.append(valids[:, ind: ind + self.S, :wind_idx])

        seq_loss = sequence_loss(coord_predictions, dep_predictions, traj_gts, vis_gts, valids_gts, 0.8, Z_WEIGHT=250)
        loss = seq_loss

        with torch.no_grad():
            query_traj_uv_pr = coord_predictions[-1][-1][:, -1, :, :]  # shape: (B, N, 2)
            query_traj_z_pr = dep_predictions[-1][-1][:, -1, :, :]  # shape: (B, N, 1)
            query_traj_gt = trajs_g[:, -1, :, :]  # shape: (B, N, 3)
            query_traj_uv_gt, query_traj_z_gt = torch.split(query_traj_gt, [2, 1], dim=-1)
            epe = torch.sum((query_traj_uv_pr - query_traj_uv_gt) ** 2, dim=-1).sqrt()
            epe = epe.view(-1)

            epe_i = (1.0 / query_traj_z_pr - 1.0 / query_traj_z_gt).abs().view(-1)

            epe_sum = epe.sum()
            px1_sum = (epe < 1).float().sum()
            px3_sum = (epe < 3).float().sum()
            px5_sum = (epe < 5).float().sum()
            epe_i_sum = epe_i.sum()
            valid1_sum = torch.ones_like(epe_sum) * epe.numel()

            dist.all_reduce(epe_sum)
            dist.all_reduce(px1_sum)
            dist.all_reduce(px3_sum)
            dist.all_reduce(px5_sum)
            dist.all_reduce(valid1_sum)

            epe = epe_sum / valid1_sum
            px1 = px1_sum / valid1_sum
            px3 = px3_sum / valid1_sum
            px5 = px5_sum / valid1_sum
            epe_i = epe_i_sum / valid1_sum

            metric_list = [
                ['epe', epe.item()],
                ['px1', px1.item()],
                ['px3', px3.item()],
                ['px5', px5.item()],
                ['epe_i', epe_i.item()]]

        return loss, metric_list

    def infer(self, model, input_list, gt_list=None, iters=4, is_train=False):

        rgbs, deps, queries = input_list

        predictions_uv, predictions_z, feat_init, train_data = model(rgbs, deps, queries, iters=iters, is_train=is_train)

        if gt_list != None:  # training mode
            loss, metric_list = self.Loss(train_data, gt_list)
            return loss, metric_list

        return predictions_uv, predictions_z, feat_init, train_data

    def training_infer(self, model, step_data, device):

        rgbs = step_data['rgbs'].to(device)  # shape: (B, T, 3, H, W)
        deps = step_data['deps'].to(device).unsqueeze(2)  # shape: (B, T, 1, H, W)
        trajs_uv = step_data['trajs_uv'].to(device)  # shape: (B, T, N, 2)
        trajs_z = step_data['trajs_z'].to(device)  # shape: (B, T, N, 1)
        vis_g = step_data['visibs'].to(device).float()  # shape: (B, T, N)
        valids = step_data['valids'].to(device).float()  # shape: (B, T, N)

        trajs = torch.cat([trajs_uv, trajs_z], dim=-1)  # shape: (B, T, N, 3)

        B, T, C, H, W = rgbs.shape  # rgbs shape: (B, T, 3, H, W)
        assert C == 3
        B, T, N, D = trajs_uv.shape  # trajs_uv shape: (B, T, N, 2)

        __, first_positive_inds = torch.max(vis_g, dim=1)  # shape: (B, N), for each B, likes (1, 0, 2, N-1, 0, ...)
        # We want to make sure that during training the model sees visible points
        # that it does not need to track just yet: they are visible but queried from a later frame
        N_rand = N // 4
        # inds of visible points in the 1st frame
        # wangbo: something wrong for batch > 1.
        nonzero_inds = [torch.nonzero(vis_g[0, :, i]) for i in range(N)]  # list(N): [shape:(x, 1), ...], x is the index number of nonzero
        rand_vis_inds = torch.cat(
            [
                nonzero_row[torch.randint(len(nonzero_row), size=(1,))]
                for nonzero_row in nonzero_inds
            ],
            dim=1,
        )  # shape: (1, N)
        first_positive_inds = torch.cat(
            [rand_vis_inds[:, :N_rand], first_positive_inds[:, N_rand:]], dim=1
        )  # shape: (B, N)
        ind_array_ = torch.arange(T, device=device)
        ind_array_ = ind_array_[None, :, None].repeat(B, 1, N)  # shape: (B, T, N)
        assert torch.allclose(
            vis_g[ind_array_ == first_positive_inds[:, None, :]],
            torch.ones_like(vis_g),
        )
        assert torch.allclose(
            vis_g[ind_array_ == rand_vis_inds[:, None, :]], torch.ones_like(vis_g)
        )

        gather = torch.gather(
            trajs, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, 3)
        )  # shape: (B, N, N, 3) means for each point, it use its first positive index to index a map, whose shape is (N, 3)
        xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)  # shape: (B, N, 3), means each initial position for each point.

        queries = torch.cat([first_positive_inds[:, :, None], xys], dim=2)  # shape: (B, N, 4), means (first time index, x, y, z) for each point.

        model = model.module

        loss, metric_list = model.infer(
            model,
            input_list=[rgbs, deps, queries],
            gt_list=[trajs, vis_g, valids],
            iters=4,
            is_train=True,
        )

        return loss, metric_list




