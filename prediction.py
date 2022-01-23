from omegaconf import OmegaConf

from assideo.config import default_config
from assideo.embeddings import Embeddings

if __name__ == '__main__':
    cfg = OmegaConf.merge(default_config(), OmegaConf.from_cli())
    embeddings = Embeddings(cfg, recreate=True)
    base_filename = cfg.get('test_filename', 'test_image.jpg')
    scores, names = embeddings.get_matches(base_filename, top_matches=20)
    for score, name in zip(scores, names):
        print(name, score.item())
