# @File: dataset.py
# @Project: SceneTracker
# @Author : wangbo
# @Time : 2024.07.04

import os.path
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.util.augmentor import LongTermSceneFlowAugmentor

data_root = '/data1/wangbo/data/'
odyssey_root = data_root + 'LSFOdyssey'
driving_root = data_root + 'LSFDriving'

class LongTermSceneFlowDataset(Dataset):
    def __init__(self, aug_params=None, track_mode='frame_first', train_mode=False):

        self.track_mode = track_mode
        self.train_mode = train_mode

        self.augmentor = None
        if aug_params is not None:
            self.augmentor = LongTermSceneFlowAugmentor(train_mode, **aug_params)

        self.meta_list = []

    def get_data_unit(self, index):
        return {}

    def __getitem__(self, index):

        data_invalid = True
        while data_invalid:

            index = index % len(self.meta_list)
            du = self.get_data_unit(index)

            if self.augmentor is not None:
                du = self.augmentor(du)

            outputs, data_invalid, index = self.prepare_output(du)

        return outputs

    def __len__(self):
        return len(self.meta_list)

    def __rmul__(self, v):
        self.meta_list = v * self.meta_list
        return self

    def prepare_output(self, du):

        rgbs = torch.from_numpy(du['rgbs']).permute(0, 3, 1, 2).float()  # shape: (T, 3, H, W)
        deps = torch.from_numpy(du['deps']).float()  # shape: (T, H, W)
        visibs = torch.from_numpy(du['visibs']).squeeze(-1)  # shape: (T, N_total)
        valids = torch.from_numpy(du['valids']).squeeze(-1)  # shape: (T, N_total)
        trajs_uv = torch.from_numpy(du['trajs_uv'])  # shape: (T, N_total, 2)
        trajs_z = torch.from_numpy(du['trajs_z'])  # shape: (T, N_total, 1)

        seq_len = du['seq_len']
        track_point_num = du['track_point_num']

        data_invalid = False
        index = None

        visibile_pts_first_frame_inds = (visibs[0]).nonzero(as_tuple=False)[:, 0]
        visibile_pts_inds = visibile_pts_first_frame_inds

        if self.train_mode:
            point_inds = torch.randperm(len(visibile_pts_inds))[: track_point_num]

            if len(point_inds) < track_point_num:
                data_invalid = True
                index = np.random.randint(0, len(self.meta_list))
        else:
            step = len(visibile_pts_inds) // track_point_num
            point_inds = list(range(0, len(visibile_pts_inds), step))[: track_point_num]

        visible_inds_sampled = visibile_pts_inds[point_inds]
        trajs_uv = trajs_uv[:, visible_inds_sampled].float()  # shape: (T, N, 2)
        trajs_z = trajs_z[:, visible_inds_sampled].float()  # shape: (T, N, 1)
        visibs = visibs[:, visible_inds_sampled]  # shape: (T, N)
        valids = valids[:, visible_inds_sampled]  # shape: (T, N)

        outputs = {
            'rgbs': rgbs,  # shape: (T, 3, H, W)
            'deps': deps,  # shape: (T, H, W)
            'visibs': visibs,  # shape: (T, N)
            'valids': valids,  # shape: (T, N)
            'trajs_uv': trajs_uv,  # shape: (T, N, 2)
            'trajs_z': trajs_z,  # shape: (T, N, 2)
        }
        if 'query_points' in du: outputs['query_points'] = torch.from_numpy(du['query_points'])
        if 'seq_name' in du: outputs['seq_name'] = du['seq_name']
        if 'intris' in du: outputs['intris'] = torch.from_numpy(du['intris'])
        if 'extris' in du: outputs['extris'] = torch.from_numpy(du['extris'])

        return outputs, data_invalid, index

class WWOdyssey(LongTermSceneFlowDataset):
    def __init__(self, aug_params=None, root=odyssey_root, seq_len=40, track_point_num=128, split='train', train_mode=False):
        super(WWOdyssey, self).__init__(aug_params, train_mode=train_mode)

        self.seq_len = seq_len
        self.track_point_num = track_point_num
        self.split = split

        data_root = f'{root}/{split}'

        seq_path_list = []
        for seq_path in glob(os.path.join(data_root, "*")):
            seq_path = seq_path.replace('\\', '/')
            if os.path.isdir(seq_path):
                seq_path_list.append(seq_path)
        seq_path_list = sorted(seq_path_list)

        for seq_path in seq_path_list:
            for sample_path in glob(os.path.join(seq_path, "*")):
                sample_path = sample_path.replace('\\', '/')
                mp4_name_path = f'{sample_path}/rgb.mp4'
                deps_name_path = f'{sample_path}/deps.npz'
                track_name_path = f'{sample_path}/track.npz'
                intris_name_path = f'{sample_path}/intris.npz'
                self.meta_list += [{
                    'mp4_name_path': mp4_name_path,
                    'deps_name_path': deps_name_path,
                    'track_name_path': track_name_path,
                    'intris_name_path': intris_name_path if split == 'test' else None,
                }]

    def read_mp4(self, name_path):
        vidcap = cv2.VideoCapture(name_path)
        frames = []
        while (vidcap.isOpened()):
            ret, frame = vidcap.read()
            if ret == False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        vidcap.release()
        return frames

    def get_data_unit(self, index):

        du = {}

        mp4_name_path = self.meta_list[index]['mp4_name_path']
        deps_name_path = self.meta_list[index]['deps_name_path']
        track_name_path = self.meta_list[index]['track_name_path']
        intris_name_path = self.meta_list[index]['intris_name_path']

        rgbs = self.read_mp4(mp4_name_path)  # list, each shape: (H, W, 3), cv2 type
        rgbs = np.stack(rgbs)  # shape: (T, H, W, 3)

        d = dict(np.load(deps_name_path, allow_pickle=True))
        if 'deps' in d:
            deps = d['deps'].astype(np.float32)   # shape: (T, 1, H, W)
        elif 'track_g' in d:
            deps = d['track_g'].astype(np.float32)  # shape: (T, 1, H, W)
        track = dict(np.load(track_name_path, allow_pickle=True))['track_g']  # shape: (T, N, 5), include: trajs, trajs_z, visibs, valids

        if intris_name_path != None:
            d = dict(np.load(intris_name_path, allow_pickle=True))
            intris = d['intris']
            extris = d['extris']

        trajs_uv = track[..., 0:2]  # shape: (T, N, 2)
        trajs_z = track[..., 2:3]  # shape: (T, N, 1)

        query_points_uv = trajs_uv[0]  # shape: (N, 2)
        query_points_z = trajs_z[0]  # shape: (N, 1)
        query_points_t = np.zeros_like(query_points_z)  # shape: (N, 1)
        query_points = np.concatenate([query_points_t, query_points_uv, query_points_z], axis=-1)  # shape: (N, 4)

        if self.track_point_num == -1:
            track_point_num = query_points.shape[0]
        else:
            track_point_num = self.track_point_num
        if self.seq_len == -1:
            seq_len = rgbs.shape[0]
        else:
            seq_len = self.seq_len

        du['rgbs'] = rgbs  # shape: (T, H, W, 3)
        du['deps'] = deps[:, 0, :, :]  # shape: (T, H, W)
        du['trajs_uv'] = trajs_uv  # shape: (T, N, 2)
        du['trajs_z'] = trajs_z  # shape: (T, N, 1)
        du['visibs'] = track[..., 3:4]  # shape: (T, N, 1)
        du['valids'] = track[..., 4:5]  # shape: (T, N, 1)
        du['seq_len'] = seq_len
        du['track_point_num'] = track_point_num
        if self.split == 'test':
            du['query_points'] = query_points  # shape: (N, 4)
            du['intris'] = intris
            du['extris'] = extris

        return du

from contextlib import contextmanager
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

def fetch_dataloader(config):
    def prepare_data(config):

        if config.stage == 'odyssey':

            aug_params = {'crop_size': config.image_size}
            dataset = WWOdyssey(aug_params, seq_len=config.seq_len, train_mode=True, track_point_num=config.track_point_num)

        return dataset

    if config.is_ddp:
        with torch_distributed_zero_first(config.rank):
            dataset = prepare_data(config)
    else:
        dataset = prepare_data(config)

    batch_size_tmp = config.batch_size // config.world_size

    dataloder = DataLoader(dataset,
                           batch_size=batch_size_tmp,
                           pin_memory=True,
                           sampler=torch.utils.data.distributed.DistributedSampler(dataset) if config.is_ddp else None,
                           num_workers=8 if config.is_ddp else 0,
                           drop_last=True)

    if config.is_master:
        print('Training with %d image pairs' % len(dataset))

    return dataloder


