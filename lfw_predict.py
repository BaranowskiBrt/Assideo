from assideo.config import load_configs
from assideo.embeddings import Embeddings
from lfw_train import LfwDataset


def main():
    cfg = load_configs('assideo/configs/lfw_config.yaml')
    dataset = LfwDataset(cfg, csv_path=cfg.test_csv, train=False)
    embeddings = Embeddings(cfg, dataset=dataset)
    if 'test_filename' in cfg:
        base_filename = cfg.test_filename
        scores, names = embeddings.get_matches(base_filename, top_matches=20)
        for score, name in zip(scores, names):
            print(name, score.item())


if __name__ == '__main__':
    main()
