import os
from torch.utils.data import Dataset
import cv2


class ImageDataset(Dataset):

    def __init__(self, root, limit=None):

        images = []

        for label in ["real", "fake"]:
            folder = os.path.join(root, label)

            for file in os.listdir(folder):
                images.append((os.path.join(folder, file), label))

        if limit:
            images = images[:limit]

        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        path, label = self.images[idx]

        img = cv2.imread(path)

        label = 1 if label == "fake" else 0

        return img, label