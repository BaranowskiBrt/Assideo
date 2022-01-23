import cv2
import torch
from torchvision import transforms

torch.set_grad_enabled(False)


class Predictor():
    def __init__(self, cfg, model, transformations=None):
        self.model = model
        self.cfg = cfg

        self.model.eval()
        self.transformations = transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ])
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state()

    def __call__(self, input):
        if isinstance(input, str):
            input = cv2.imread(input)
        if self.transformations:
            input = self.transformations(input)
        if not isinstance(input, torch.Tensor):
            input = self.transform(input)
        input = input.unsqueeze(0)
        return self.model(input)

    def load_state(self):
        save_type = str(self.cfg.save_type).lower().strip()
        if save_type in ['parameters', 'weights']:
            self.model.load_state_dict(torch.load(self.cfg.saved_model_path,
                                                  map_location=self.device),
                                       strict=False)
        elif save_type == 'model':
            self.model = torch.load(self.cfg.saved_model_path,
                                    map_location=self.device)
        else:
            raise ValueError(f"Saving type {save_type} does not exist.")
