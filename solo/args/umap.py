import argparse

from solo.args.dataset import custom_dataset_args, dataset_args


def parse_args_umap() -> argparse.Namespace:
    """Parses arguments for offline UMAP.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    # add knn args
    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=10)

    # noise args
    parser.add_argument("--snr_db", type=int, default=None)

    # seed args
    parser.add_argument("--rand_seed", type=int, default=5)
    
    # add shared arguments
    dataset_args(parser)
    custom_dataset_args(parser)

    # parse args
    args = parser.parse_args()

    return args
