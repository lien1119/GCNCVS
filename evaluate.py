import re
import time
import data
import init
import torch
import metric
import logging
from pathlib import Path
from GCNCVS.GVS import GCN_multi_feature


logger = logging.getLogger()

def get_GCN_multi_feature(mv_feature,residual_feature,partition_feature,dcavg_feature,qp_feature,**kwargs):
    return GCN_multi_feature(mv_feature,residual_feature,partition_feature,dcavg_feature,qp_feature)

def evaluate(model, val_loader):

    model.eval()
    stats = data.AverageMeter('fscore','spearmanr','kendalltau')
    
    with torch.no_grad():
        
        for test_key, mvData,residualData,partitionData,dcavgData,qpData, cps, n_frames, nfps, picks, user_summary, video_name in val_loader:

            
            output = model(mvData,residualData,partitionData,dcavgData,qpData)
            output = output.squeeze(0).cpu().numpy()
            
            
            pred_summ = metric.generate_summary(
                output, cps, n_frames, nfps, picks)
            
            

            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            fscore = metric.get_summ_f1score(video_name, pred_summ, user_summary, eval_metric)

            if 'summe' in test_key:
                summe_mat = data.load_summe_mat('matfile/summe/GT/')
                all_frame_scores = metric.get_all_frame_score(output,n_frames,picks)
                for item in summe_mat:
                    if(item['nframes'] == n_frames):
                        user_anno = item['user_anno'].T
                spearmanr = metric.get_correlation('spearmanr',user_anno,all_frame_scores)
                kendalltau = metric.get_correlation('kendalltau',user_anno,all_frame_scores)

            if 'tvsum' in test_key:
                tvsum_mat = data.load_tvsum_mat('matfile/tvsum/ydata-tvsum50.mat')
                match = re.search(r'\d+',video_name)
                number = int(match.group())
                user_anno = tvsum_mat[number-1]['user_anno'].T
                all_frame_scores = metric.get_all_frame_score(output,n_frames,picks)
                spearmanr = metric.get_correlation('spearmanr',user_anno,all_frame_scores)
                kendalltau = metric.get_correlation('kendalltau',user_anno,all_frame_scores)

            stats.update(fscore=fscore,spearmanr=spearmanr,kendalltau=kendalltau)
    return fscore, spearmanr, kendalltau

def main():
    args = init.get_arguments()

    init.init_logger(args.model_dir, args.log_file)

    logger.info(vars(args))
    model = get_GCN_multi_feature(**vars(args))
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters: {total_params}")
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data.load_yaml(split_path)

        stats = data.AverageMeter('fscore','spearmanr','kendalltau')

        for split_idx, split in enumerate(splits):
            ckpt_path = data.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path),
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

            val_set = data.VideoDataset(split['test_keys'])
            val_loader = data.DataLoader(val_set, shuffle=False)

            fscore,spearmanr, kendalltau = evaluate(model, val_loader)
            stats.update(fscore=fscore,spearmanr=spearmanr,kendalltau=kendalltau)

            logger.info(f'{split_path.stem} split {split_idx} F-score: {fscore:.4f} spearmanr: {spearmanr:.5f} kendalltau:{kendalltau:.5f}')

        logger.info(f'{split_path.stem}: F-score: {stats.fscore:.4f} {fscore:.4f} spearmanr: {stats.spearmanr:.5f} kendalltau:{stats.kendalltau:.5f}')



if __name__ == '__main__':
    main()