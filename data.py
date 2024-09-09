import os
import h5py
import yaml
import torch
import random
import hdf5storage
import numpy as np
import scipy.io as sio
from os import PathLike
from pathlib import Path
from typing import Any, List, Dict
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

def load_summe_mat(dirname):
    mat_list = os.listdir(dirname)

    data_list = []
    for mat in mat_list:
        data = sio.loadmat(os.path.join(dirname, mat))

        item_dict = {
        'video': mat[:-4],
        'length': data['video_duration'],
        'nframes': data['nFrames'],
        'user_anno': data['user_score'],
        'gt_score': data['gt_score']
        }
        
        data_list.append((item_dict))
    
    return data_list

def load_tvsum_mat(filename):
    data = hdf5storage.loadmat(filename,variable_names=['tvsum50'])
    data = data['tvsum50'].ravel()
    
    data_list = []
    for item in data:
        video, category, title, length, nframes, user_anno, gt_score = item
        
        item_dict = {
        'video': video[0, 0],
        'category': category[0, 0],
        'title': title[0, 0],
        'length': length[0, 0],
        'nframes': nframes[0, 0],
        'user_anno': user_anno,
        'gt_score': gt_score
        }
        
        data_list.append((item_dict))
    
    return data_list

class VideoDataset(object):
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.datasets = self.get_datasets(keys)

    def __getitem__(self, index):
        key = self.keys[index]
        video_path = Path(key)
        dataset_name = str(video_path.parent)
        video_name = video_path.name
        video_file = self.datasets[dataset_name][video_name]

        mv = video_file['mv'][...].astype(np.float32)
        residual = video_file['residual'][...].astype(np.float32)
        partition = video_file['partition'][...].astype(np.float32)
        dcavg = video_file['dcavg'][...].astype(np.float32)
        qp = video_file['qp'][...].astype(np.float32)

        gtscore = video_file['gtscore'][...].astype(np.float32)
        cps = video_file['change_points'][...].astype(np.int32)
        n_frames = video_file['n_frames'][...].astype(np.int32)
        nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)
        user_summary = None
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)

        gtscore -= gtscore.min()
        gtscore /= gtscore.max()

        mv = torch.tensor(
           mv, dtype=torch.float32).to('cuda')
        residual = torch.tensor(
           residual, dtype=torch.float32).to('cuda')
        partition = torch.tensor(
           partition, dtype=torch.float32).to('cuda')
        dcavg = torch.tensor(
            dcavg, dtype=torch.float32).to('cuda')
        qp = torch.tensor(
            qp, dtype=torch.float32).to('cuda')   
        

        gtscore = torch.tensor(
            gtscore, dtype=torch.float32).to('cuda')

        mv_edge_index = knn_graph(mv, k=30, cosine=True)
        residual_edge_index = knn_graph(residual, k=30, cosine=True)
        partition_edge_index = knn_graph(partition, k=30, cosine=True)
        dcavg_edge_index = knn_graph(dcavg, k=30, cosine=True)
        qp_edge_index = knn_graph(qp, k=30, cosine=True)

        mvData = Data(x=mv,edge_index=mv_edge_index, y=gtscore)
        residualData = Data(x=residual,edge_index=residual_edge_index, y=gtscore)
        partitionData = Data(x=partition,edge_index=partition_edge_index, y=gtscore)
        dcavgData = Data(x=dcavg,edge_index=dcavg_edge_index, y=gtscore)
        qpData = Data(x=qp,edge_index=qp_edge_index, y=gtscore)

        return key, mvData,residualData,partitionData,dcavgData,qpData, cps, n_frames, nfps, picks, user_summary, video_name
    
    def __len__(self):
        return len(self.keys)

    @staticmethod
    def get_datasets(keys: List[str]) -> Dict[str, h5py.File]:
        dataset_paths = {str(Path(key).parent) for key in keys}
        datasets = {path: h5py.File(path, 'r') for path in dataset_paths}
        return datasets


class DataLoader(object):
    def __init__(self, dataset: VideoDataset, shuffle: bool):
        self.dataset = dataset
        self.shuffle = shuffle
        self.data_idx = list(range(len(self.dataset)))

    def __iter__(self):
        self.iter_idx = 0
        if self.shuffle:
            random.shuffle(self.data_idx)
        return self

    def __next__(self):
        if self.iter_idx == len(self.dataset):
            raise StopIteration
        curr_idx = self.data_idx[self.iter_idx]
        batch = self.dataset[curr_idx]
        self.iter_idx += 1
        return batch


class AverageMeter(object):
    def __init__(self, *keys: str):
        self.totals = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self._check_attr(key)
            self.totals[key] += value
            self.counts[key] += 1

    def __getattr__(self, attr: str) -> float:
        self._check_attr(attr)
        total = self.totals[attr]
        count = self.counts[attr]
        return total / count if count else 0.0

    def _check_attr(self, attr: str) -> None:
        assert attr in self.totals and attr in self.counts


def get_ckpt_dir(model_dir: PathLike) -> Path:
    return Path(model_dir) / 'checkpoint'


def get_ckpt_path(model_dir: PathLike,
                  split_path: PathLike,
                  split_index: int) -> Path:
    split_path = Path(split_path)
    return get_ckpt_dir(model_dir) / f'{split_path.name}.{split_index}.pt'


def load_yaml(path: PathLike) -> Any:
    with open(path) as f:
        obj = yaml.safe_load(f)
    return obj


def dump_yaml(obj: Any, path: PathLike) -> None:
    with open(path, 'w') as f:
        yaml.dump(obj, f)
