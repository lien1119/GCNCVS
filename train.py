import logging
import init
import data
from pathlib import Path
from GCNCVS.train import train

logger = logging.getLogger()

def main():
    args = init.get_arguments()
    init.init_logger(args.model_dir, args.log_file)

    logger.info(vars(args))

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data.get_ckpt_dir(model_dir).mkdir(parents=True, exist_ok=True)


    data.dump_yaml(vars(args), model_dir/ 'args.yml')

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data.load_yaml(split_path)

        results = {}
        stats = data.AverageMeter('fscore','spearmanr','kendalltau','epochtime')
        for split_idx, split in enumerate(splits):
            logger.info(f'Start training on {split_path.stem}: split {split_idx}')
            ckpt_path = data.get_ckpt_path(model_dir, split_path, split_idx)
            fscore, spearmanr, kendalltau,epoch_time = train(args, split, ckpt_path)
            stats.update(fscore=fscore,spearmanr=spearmanr,kendalltau=kendalltau,epochtime=epoch_time)
            results[f'split{split_idx}'] = float(fscore)

        results['mean'] = float(stats.fscore)
        data.dump_yaml(results, model_dir / f'{split_path.stem}.yml')

        logger.info(f'Training done on {split_path.stem}. F-score: {stats.fscore:.4f} spearmanr: {stats.spearmanr:.5f} kendalltau:{stats.kendalltau:.5f} sec/epoch:{stats.epochtime:.5f}')

if __name__ == '__main__':
    main()
