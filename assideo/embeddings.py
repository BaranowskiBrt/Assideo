from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity
from torchvision import transforms as T
from tqdm import tqdm

from .dataset import RetrievalDataset, collate_fn
from .model import BaseModel
from .predictor import Predictor


class Embeddings:
    def __init__(self,
                 cfg,
                 predictor=None,
                 dataset=None,
                 embeddings=None,
                 recreate=False):
        self.cfg = cfg

        if predictor:
            self.predictor = predictor
        else:
            model = BaseModel(self.cfg)
            self.predictor = Predictor(self.cfg, model)

        self.cos_similarity = CosineSimilarity(dim=-1, eps=1e-8)
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(cfg.mean, cfg.std),
        ])

        if embeddings:
            embedding_dict = embeddings
        elif not recreate and Path(cfg.get('embeddings_path', '')).is_file():
            embedding_dict = torch.load(cfg.embeddings_path)
        else:
            dataset = dataset or RetrievalDataset(cfg, train=False)
            embedding_dict = self.create_embeddings(dataset)

        self.img_paths, self.embeddings = embedding_dict[
            'relative_path'], embedding_dict['embeddings']

    def create_embeddings(self, dataset):

        embeddings = []
        img_paths = []
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=collate_fn,
        )
        for data in tqdm(loader):
            output = self.predictor(data['image'][0])
            output = output.double()
            embeddings.append(output)
            img_paths.append(data['relative_path'][0])
        embeddings = torch.stack(embeddings)
        embedding_dict = {'relative_path': img_paths, 'embeddings': embeddings}
        torch.save(embedding_dict, self.cfg.embeddings_path)
        return embedding_dict

    def get_similar_embeddings(self, embedding):
        return self.cos_similarity(embedding, self.embeddings)

    def get_matches(self, image_path, top_matches=5):
        img = cv2.imread(image_path)
        base_output = self.predictor(self.transforms(img))

        matched = self.get_similar_embeddings(base_output).squeeze()
        sort, indices = matched.sort(descending=True)
        cat_paths = np.array(self.img_paths)[indices]
        return sort[:top_matches], cat_paths[:top_matches]

    def accuracy_score(self, threshold=0.5):
        TP = FP = TN = FN = 0
        for i in tqdm(range(len(self.embeddings))):
            ref_cat = Path(self.img_paths[i]).parent
            scores = self.get_similar_embeddings(self.embeddings[i]).squeeze()

            for j, (cat_path, score) in enumerate(zip(self.img_paths, scores)):
                if j == i:
                    continue
                cat = Path(cat_path).parent
                if score >= threshold:
                    if ref_cat == cat:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if ref_cat != cat:
                        TN += 1
                    else:
                        FN += 1

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        return precision, recall, accuracy
