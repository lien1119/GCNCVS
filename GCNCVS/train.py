import os
import time
import data
import h5py
import torch
import logging
import numpy as np
from torch import nn
from evaluate import evaluate
from GCNCVS.GVS import GCN_multi_feature
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
logger = logging.getLogger()


def train(args, split,save_path):
    model = GCN_multi_feature(args.mv_feature,args.residual_feature,args.partition_feature,args.dcavg_feature,args.qp_feature,1024)
   
    model = model.to(args.device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        parameters, lr=args.lr, weight_decay=args.weight_decay)
    
    criterion = nn.MSELoss()
    
    train_set = data.VideoDataset(split['train_keys'])
    train_loader = data.DataLoader(train_set, shuffle=True)

    val_set = data.VideoDataset(split['test_keys'])
    val_loader = data.DataLoader(val_set, shuffle=False)
    
    max_val_fscore = -1
    max_val_spearmanr = -1
    max_val_kendalltau = -1
    avg_sec = []
    for epoch in range(args.max_epoch):
        start_time = time.time()
        model.train()
        stats = data.AverageMeter('loss')

        for key, mvData,residualData,partitionData,dcavgData,qpData, cps, n_frames, nfps, picks, _, name in train_loader:

            output = model(mvData,residualData,partitionData,dcavgData,qpData)
            
            loss = criterion(output.squeeze(0), qpData.y.squeeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(loss=loss.item())
        end_time = time.time()

        val_fscore, val_spearmanr, val_kendalltau  = evaluate(model, val_loader)
        
        if max_val_spearmanr <= val_spearmanr:
            max_val_spearmanr = val_spearmanr

        if max_val_kendalltau <= val_kendalltau:
            max_val_kendalltau = val_kendalltau

        if max_val_fscore <= val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))
            
        avg_sec.append(end_time-start_time)
        logger.info(f'Epoch: {epoch}/{args.max_epoch} '
                    f'Loss: {stats.loss:.4f} '
                    f'F-score/max: {val_fscore:.4f}/{max_val_fscore:.4f} '
                    f'spe/max: {val_spearmanr:.4f}/{max_val_spearmanr:.4f} '
                    f'ken/max: {val_kendalltau:.4f}/{max_val_kendalltau:.4f} ')
    print(f"sec/epoch: {np.mean(avg_sec):.4f}")
    return max_val_fscore, max_val_spearmanr, max_val_kendalltau, np.mean(avg_sec)