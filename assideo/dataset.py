from pathlib import Path, PurePath

import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T


def collate_fn(data):
    assert all([instance.keys() == data[0].keys() for instance in data[1:]
                ]), 'All instances in a batch should have the same keys'
    return {
        key: [instance[key] for instance in data]
        for key in data[0].keys()
    }


class RetrievalDataset(Dataset):
    def __init__(self, cfg, train=True, extension='jpg', augmentations=None):
        self.extension = extension.strip('. ')
        self.img_dir = Path(
            cfg.train_image_dir if train else cfg.test_image_dir)

        if not augmentations and train:
            augmentations = T.Compose([T.RandomHorizontalFlip(p=0.5)])
        self.images = []
        categories = set()
        for img_path in self.img_dir.rglob(f'*.{extension}'):
            relative_path = PurePath(img_path).relative_to(self.img_dir)
            if not self.inclusion_fn(relative_path):
                continue
            img_cat = self.generate_category(relative_path)
            self.images.append({
                'relative_path': str(relative_path),
                'category_name': img_cat
            })
            categories.add(img_cat)
        self.cat_names = sorted(categories)
        self.augmentations = augmentations
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(cfg.mean, cfg.std),
        ])

    def generate_category(self, path):
        return path.parent

    def inclusion_fn(self, path):
        return True

    def __getitem__(self, index):
        img_dict = self.images[index]
        img = cv2.imread(str(Path(self.img_dir, img_dict['relative_path'])))

        img = self.transform(img)
        if self.augmentations:
            img = self.augmentations(img)
        return {
            'image': img,
            'category_id': self.cat_names.index(img_dict['category_name']),
            **img_dict
        }

    def __len__(self):
        return len(self.images)

    def get_category_count(self):
        return len(self.cat_names)
