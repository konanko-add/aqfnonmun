import json, os
from PIL import Image
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.image_dir = image_dir
        self.items = []
        for ann in data['annotations']:
            img = next(img for img in data['images'] if img['id'] == ann['image_id'])
            path = os.path.join(image_dir, img['file_name'])
            self.items.append((path, ann['caption']))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, caption = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, caption
