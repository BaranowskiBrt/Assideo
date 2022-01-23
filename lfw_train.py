import pandas as pd

from assideo.dataset import RetrievalDataset
from assideo.config import load_configs
from assideo.model import TrainingModel
from assideo.trainer import BaseTrainer


class LfwDataset(RetrievalDataset):
    def __init__(self, *args, csv_path, **kwargs):
        data = pd.read_csv(csv_path)
        self.allowed_categories = set(data['name'])
        super().__init__(*args, **kwargs)

    def inclusion_fn(self, path):
        return path.parts[-2] in self.allowed_categories


def main():
    cfg = load_configs('assideo/configs/lfw_config.yaml')
    dataset = LfwDataset(cfg, csv_path=cfg.train_csv, train=True)
    model = TrainingModel(cfg, dataset.get_category_count())
    trainer = BaseTrainer(cfg, model, dataset)
    trainer.train(save=True)


if __name__ == '__main__':
    main()