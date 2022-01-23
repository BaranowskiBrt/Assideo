from os.path import join

import numpy as np
from sklearn.metrics import roc_curve, auc
from torch.nn import CosineSimilarity
from tqdm import tqdm

from .model import BaseModel
from .predictor import Predictor


def get_similarities(pairs, name_gen, test_image_dir, predictor,
                     cos_similarity):
    similarities = []
    for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
        img1, img2 = name_gen(row)
        img1, img2 = join(test_image_dir, img1), join(test_image_dir, img2)

        embed1, embed2 = predictor(img1), predictor(img2)
        similarity = cos_similarity(embed1, embed2)
        similarities.append(similarity)
    return similarities


def get_metrics(cfg,
                match_pairs,
                mismatch_pairs,
                match_name_gen,
                mismatch_name_gen,
                model=None):
    if not model:
        model = BaseModel(cfg)
        predictor = Predictor(cfg, model)
    cos_similarity = CosineSimilarity(dim=-1, eps=1e-8)

    matched_scores = get_similarities(match_pairs, match_name_gen,
                                      cfg.test_image_dir, predictor,
                                      cos_similarity)

    mismatched_scores = get_similarities(mismatch_pairs, mismatch_name_gen,
                                         cfg.test_image_dir, predictor,
                                         cos_similarity)
    fpr, tpr, thresholds = roc_curve(
        len(matched_scores) * [1] + len(matched_scores) * [0],
        matched_scores + mismatched_scores)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f'Optimal threshold: {optimal_threshold}')
    tp = len([1 for score in matched_scores if score >= optimal_threshold])
    fn = len(matched_scores) - tp
    fp = len([1 for score in mismatched_scores if score >= optimal_threshold])
    tn = len(mismatched_scores) - fp
    return {
        'precision': tp / (tp + fp),
        'recall': tp / (tp + fn),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'auc': auc(fpr, tpr),
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'optimal_threshold': optimal_threshold
    }
