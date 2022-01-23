from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from assideo.config import load_configs
from assideo.metrics import get_metrics


def name_generator(name, number):
    return path.join(name, f"{name}_{str(number).zfill(4)}.jpg")


def main():
    cfg = load_configs('assideo/configs/lfw_config.yaml')
    cfg.weights = [
        '../assideo_files/weights/softmax.pt',
        '../assideo_files/weights/cosface.pt',
        '../assideo_files/weights/arcface.pt',
    ]
    label_caption = [
        'Softmax',
        'Cosface',
        'Arcface',
    ]
    embedding_sizes = len(label_caption) * [512]

    match_pairs = pd.read_csv(cfg.match_pairs)
    mismatch_pairs = pd.read_csv(cfg.mismatch_pairs)
    match_path_generator = lambda x: (name_generator(x['name'], x[
        'imagenum1']), name_generator(x['name'], x['imagenum2']))
    mismatch_path_generator = lambda x: (name_generator(
        x['name'], x['imagenum1']), name_generator(x['name.1'], x['imagenum2'])
                                         )
    for weight, label_caption, embedding_size in zip(cfg.weights,
                                                     label_caption,
                                                     embedding_sizes):
        cfg.saved_model_path = weight
        cfg.embedding_length = embedding_size
        metrics = get_metrics(cfg, match_pairs, mismatch_pairs,
                              match_path_generator, mismatch_path_generator)
        print(
            f"{label_caption}: Precision: {metrics['precision']}, Recall: {metrics['recall']}, Accuracy: {metrics['accuracy']}, AUC: {metrics['auc']}"
        )

        plt.plot(metrics['fpr'],
                 metrics['tpr'],
                 label=f"{label_caption}, {metrics['accuracy']} accuracy")
        optimal_idx = np.argmax(metrics['tpr'] - metrics['fpr'])
        plt.plot(metrics['fpr'][optimal_idx],
                 metrics['tpr'][optimal_idx],
                 marker='o',
                 markersize=3,
                 color="red")

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
