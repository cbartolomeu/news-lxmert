from src.lxmert import LXMERT

from src.datasets.nytimes_dataset import NYTimesDataset, SimpleNYTimesDataset
from src.utils.params import params


def main():
    args = params()
    print(args)

    max_batch_size = 64
    accum_iter = int(args.batchSize / max_batch_size)
    print(f"Batch size: {max_batch_size}*{accum_iter}={args.batchSize}")

    # Create datasets
    train_dset = NYTimesDataset(args.trainDsDir, args.featsDir, args.trainIndex, args.mode)
    eval_dset = NYTimesDataset(args.validDsDir, args.featsDir, args.validIndex, args.mode)
    test_dset = NYTimesDataset(args.testDsDir, args.featsDir, args.testIndex, args.mode)

    # Create datasets for metrics
    train_metrics_dset = SimpleNYTimesDataset(args.trainDsDir, args.featsDir, args.trainIndex, args.mode)
    eval_metrics_dset = SimpleNYTimesDataset(args.validDsDir, args.featsDir, args.validIndex, args.mode)
    test_metrics_dset = SimpleNYTimesDataset(args.testDsDir, args.featsDir, args.testIndex, args.mode)

    # Create model
    model = LXMERT(train_dset=train_dset, eval_dset=eval_dset, test_dset=test_dset,
                   train_metrics_dset=train_metrics_dset, eval_metrics_dset=eval_metrics_dset,
                   test_metrics_dset=test_metrics_dset, output=args.output, max_seq_length=args.maxSeqLen, max_faces=10,
                   batch_size=max_batch_size, masked_lm_ratio=args.maskedLmRatio, ent_mask_ratio=0.30,
                   masked_feats_ratio=args.maskedFeatsRatio, load=args.load, load_lxmert=args.loadLxmert,
                   from_scratch=args.fromScratch, use_entities=args.entities, use_faces=args.faces)

    if args.test:
        model.test()
    else:
        model.train(epochs=args.epochs, lr=args.lr, warmup_ratio=args.warmupRatio, accum_iter=accum_iter)


if __name__ == '__main__':
    main()
