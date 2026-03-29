"""
FaceForensics++ dataset loader with MTCNN face detection.

Provides:
- FaceForensicsDataset: PyTorch Dataset for FaceForensics++ images
- download_faceforensics_dataset: Download dataset via kagglehub
- get_faceforensics_transforms: Standard transforms for face images
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import kagglehub

# Try to import MTCNN from facenet-pytorch
try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

# Optional OpenCV import for image loading
try:
    import cv2
except ImportError:
    cv2 = None


def download_faceforensics_dataset():
    """
    Download FaceForensics++ dataset from Kaggle via kagglehub.

    Returns:
        str: Path to the downloaded dataset directory.
    """
    path = kagglehub.dataset_download("hungle3401/faceforensics")
    print(f"Dataset downloaded to: {path}")
    return path


def get_faceforensics_transforms():
    """
    Get standard transforms for FaceForensics++ face images.

    Returns appropriate ImageNet-normalized transforms for EfficientNet-B0:
    - Resize to 224x224 (EfficientNet input size)
    - Convert to tensor
    - Normalize with ImageNet mean/std

    Returns:
        transforms.Compose: Composed transforms for face images.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class FaceForensicsDataset(Dataset):
    """
    PyTorch Dataset for FaceForensics++ deepfake detection.

    Loads images from FaceForensics++ dataset structure:
    - Real images: original_sequences/{compression}/images/{video_id}/*.png
    - Fake images: manipulated_sequences/Deepfakes/{compression}/images/{video_id}/*.png

    Uses MTCNN for face detection in __getitem__, with fallback to
    direct resize if no face is detected.

    Binary classification:
    - Real = 0
    - Fake = 1

    Args:
        root (str): Root directory of the FaceForensics++ dataset.
        transform (transforms.Compose, optional): Transform to apply to face crops.
            Defaults to get_faceforensics_transforms().
        limit (int, optional): Maximum number of samples to load (for development).
            Defaults to None (load all).
        compression (str): Compression level folder name (e.g., 'c23', 'c40').
            Defaults to 'c23' (visually lossless).
        device (str): Device to run MTCNN on ('cuda' or 'cpu').
            Defaults to 'cuda' if available, else 'cpu'.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        limit: int = None,
        compression: str = "c23",
        device: str = None,
    ):
        self.root = root
        self.transform = transform or get_faceforensics_transforms()
        self.limit = limit
        self.compression = compression

        # Set device for MTCNN
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize MTCNN for face detection
        if MTCNN is not None:
            self.mtcnn = MTCNN(
                image_size=224,
                margin=20,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device,
                keep_all=False,  # Keep only largest face
            )
        else:
            self.mtcnn = None

        # Load image paths and labels
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Load all image paths from the dataset directory.

        Returns:
            list: List of (image_path, label) tuples where label is 0 (real) or 1 (fake).
        """
        samples = []

        # Load real images from original_sequences
        real_dir = os.path.join(
            self.root, "original_sequences", "youtube", self.compression, "images"
        )
        if os.path.exists(real_dir):
            samples.extend(self._collect_images_from_structure(real_dir, label=0))

        # Load fake images from manipulated_sequences/Deepfakes
        fake_dir = os.path.join(
            self.root, "manipulated_sequences", "Deepfakes", self.compression, "images"
        )
        if os.path.exists(fake_dir):
            samples.extend(self._collect_images_from_structure(fake_dir, label=1))

        # Check if dataset structure exists; if not, try flat structure
        if not samples:
            samples = self._load_flat_structure()

        # Apply limit if specified
        if self.limit:
            samples = samples[: self.limit]

        return samples

    def _collect_images_from_structure(self, base_dir: str, label: int):
        """
        Recursively collect image paths from video folder structure.

        FaceForensics++ has structure: images/{video_id}/*.png

        Args:
            base_dir (str): Base directory containing video folders.
            label (int): Label for all images in this directory (0=real, 1=fake).

        Returns:
            list: List of (image_path, label) tuples.
        """
        samples = []

        if not os.path.exists(base_dir):
            return samples

        for video_id in os.listdir(base_dir):
            video_dir = os.path.join(base_dir, video_id)
            if not os.path.isdir(video_dir):
                continue

            for filename in os.listdir(video_dir):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    samples.append((os.path.join(video_dir, filename), label))

        return samples

    def _load_flat_structure(self):
        """
        Load images from flat directory structure (alternative layout).

        Expects: root/{real,fake}/*.png

        Returns:
            list: List of (image_path, label) tuples.
        """
        samples = []

        for label_name, label in [("real", 0), ("fake", 1)]:
            label_dir = os.path.join(self.root, label_name)
            if os.path.exists(label_dir):
                for filename in os.listdir(label_dir):
                    if filename.endswith((".png", ".jpg", ".jpeg")):
                        samples.append((os.path.join(label_dir, filename), label))

        return samples

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Applies MTCNN face detection and crops the face region.
        Falls back to resizing the original image if face detection fails.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image_tensor, label) where image_tensor is a transformed face crop.
        """
        image_path, label = self.samples[idx]

        # Load image
        try:
            # Try PIL first (works better with MTCNN)
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # Fallback to OpenCV if PIL fails
            if cv2 is not None:
                img_array = cv2.imread(image_path)
                if img_array is not None:
                    image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                else:
                    # Cannot load image - return black image
                    image = Image.new("RGB", (224, 224))
            else:
                # No OpenCV - return black image
                image = Image.new("RGB", (224, 224))

        # Apply MTCNN face detection if available
        if self.mtcnn is not None:
            try:
                # MTCNN expects batch dimension
                face_crop = self.mtcnn(image)

                if face_crop is not None:
                    # MTCNN returns tensor in [-1, 1], convert to PIL for transforms
                    # Normalize back to [0, 1] then convert
                    face_crop = (face_crop + 1) / 2
                    face_crop = transforms.ToPILImage()(face_crop.squeeze(0).cpu())
                else:
                    # No face detected - resize original
                    face_crop = image.resize((224, 224), Image.BILINEAR)

            except Exception:
                # MTCNN failed - fallback to resize
                face_crop = image.resize((224, 224), Image.BILINEAR)
        else:
            # MTCNN not available - resize original
            face_crop = image.resize((224, 224), Image.BILINEAR)

        # Apply transforms
        if self.transform:
            face_tensor = self.transform(face_crop)
        else:
            face_tensor = transforms.ToTensor()(face_crop)

        return face_tensor, label
