import argparse


def embedding_params():
    parser = argparse.ArgumentParser(description="Use Lxmert to extract embeddings from NYTimes dataset.")
    parser.add_argument("--dsDir", type=str, help="Directory where are all documents", required=True)
    parser.add_argument("--index", type=str,
                        help="Pickle file that contains all documents in a dictionary organization",
                        required=True)
    parser.add_argument("--featsDir", type=str, help="Directory where are all image features", required=True)
    parser.add_argument("--output", type=str, help="Directory to save model checkpoints", required=True)

    parser.add_argument("--load", type=str, help="Model to load", required=False)
    parser.add_argument("--loadLxmert", type=str, help="Model to load without answer heads", required=False)
    parser.add_argument("--fromScratch", help="Load model from scratch without VQA train", required=False,
                        action="store_true")

    parser.add_argument("--batchSize", type=int, help="Dataloader batch sizer", required=True)
    parser.add_argument("--mode", type=int, help="Text mode", required=True)

    parser.add_argument("--maxSeqLen", type=int, help="Max token sequence length for text", required=True)

    parser.add_argument("--test", help="Only log test metrics", required=False, action="store_true")
    parser.add_argument("--entities", type=str2bool, help="Use entities for training", required=True)
    parser.add_argument("--faces", type=str2bool, help="Use faces for training", required=True)

    return parser.parse_args()


def params():
    parser = argparse.ArgumentParser(description="Train Lxmert with the NYTimes dataset.")
    parser.add_argument("--trainDsDir", type=str, help="Directory where are all training documents", required=True)
    parser.add_argument("--trainIndex", type=str,
                        help="Pickle file that contains all train documents in a dictionary organization",
                        required=True)
    parser.add_argument("--validDsDir", type=str, help="Directory where are all training documents", required=True)
    parser.add_argument("--validIndex", type=str,
                        help="Pickle file that contains all validation documents in a dictionary organization",
                        required=True)
    parser.add_argument("--testDsDir", type=str, help="Directory where are all test documents", required=True)
    parser.add_argument("--testIndex", type=str,
                        help="Pickle file that contains all test documents in a dictionary organization",
                        required=True)
    parser.add_argument("--featsDir", type=str, help="Directory where are all image features", required=True)
    parser.add_argument("--output", type=str, help="Directory to save model checkpoints", required=True)

    parser.add_argument("--load", type=str, help="Model to load", required=False)
    parser.add_argument("--loadLxmert", type=str, help="Model to load without answer heads", required=False)
    parser.add_argument("--fromScratch", help="Load model from scratch without VQA train", required=False,
                        action="store_true")

    parser.add_argument("--epochs", type=int, help="Number of epochs to train", required=True)
    parser.add_argument("--batchSize", type=int, help="Dataloader batch sizer", required=True)
    parser.add_argument("--lr", type=float, help="Learning rate", required=True)
    parser.add_argument("--warmupRatio", type=float, help="Warmup ratio", required=True)
    parser.add_argument("--mode", type=int, help="Text mode", required=True)

    parser.add_argument("--maskedLmRatio", type=float, help="Masked ratio for language training tasks", required=True)
    parser.add_argument("--maskedFeatsRatio", type=float, help="Masked ratio for visual training tasks", required=True)

    parser.add_argument("--maxSeqLen", type=int, help="Max token sequence length for text", required=True)

    parser.add_argument("--test", help="Only log test metrics", required=False, action="store_true")
    parser.add_argument("--entities", type=str2bool, help="Use entities for training", required=True)
    parser.add_argument("--faces", type=str2bool, help="Use faces for training", required=True)

    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
