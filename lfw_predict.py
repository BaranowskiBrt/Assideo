import os

from assideo.config import load_configs
from assideo.embeddings import Embeddings
from lfw_train import LfwDataset


def main():
    cfg = load_configs('assideo/configs/lfw_config.yaml')
    dataset = LfwDataset(cfg, csv_path=cfg.test_csv, train=False)
    embeddings = Embeddings(cfg, dataset=dataset)
    if 'test_filename' in cfg:
        base_filename = cfg.test_filename
        scores, names = embeddings.get_matches(base_filename, top_matches=31)
        for i, (score, name) in enumerate(zip(scores[1:], names[1:])):
            print(
                f"#{str(i + 1).ljust(3)} Image: {os.path.split(name)[-1].ljust(30)}",
                f"     score: {round(score.item(), 4)}")


if __name__ == '__main__':
    main()
