"""
FaceForensics++ dataset loader with MTCNN face detection.

Provides:
- FaceForensicsDataset: PyTorch Dataset for FaceForensics++ images
- download_faceforensics_dataset: Download dataset via kagglehub
- get_faceforensics_transforms: Standard transforms for face images
- precompute_face_crops: Pre-compute face crops for faster training

Performance optimizations:
- Pre-compute face crops once and cache them (avoids MTCNN in __getitem__)
- Memory-mapped storage for face crops
- Batch processing for MTCNN
"""

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import kagglehub
import numpy as np

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


def get_faceforensics_transforms(include_resize=False):
    """
    Get transforms for FaceForensics++ face images.

    Args:
        include_resize: If True, include Resize (for raw images).
                       If False, skip resize (for pre-cropped 224x224 images).

    Returns ImageNet-normalized transforms for EfficientNet-B0:
    - Optional resize to 224x224
    - Convert to tensor
    - Normalize with ImageNet mean/std

    Returns:
        transforms.Compose: Composed transforms.
    """
    transform_list = []
    if include_resize:
        transform_list.append(transforms.Resize((224, 224)))
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(transform_list)


def precompute_face_crops(
    dataset_root: str,
    cache_dir: str = None,
    compression: str = "c23",
    batch_size: int = 32,
    device: str = None,
):
    """
    Pre-compute face crops using MTCNN and cache to disk.

    This is the recommended way to use FaceForensicsDataset for training,
    as running MTCNN in __getitem__ is extremely slow.

    Args:
        dataset_root: Root directory of FaceForensics++ dataset.
        cache_dir: Directory to store cached crops. Defaults to {dataset_root}/crops/.
        compression: Compression level (c23 or c40).
        batch_size: Batch size for MTCNN processing.
        device: Device for MTCNN ('cuda' or 'cpu'). Defaults to cuda if available.

    Returns:
        str: Path to the cache directory containing pre-computed crops.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cache_dir is None:
        cache_dir = os.path.join(dataset_root, "crops")

    os.makedirs(cache_dir, exist_ok=True)

    manifest_path = os.path.join(cache_dir, "manifest.json")

    if os.path.exists(manifest_path):
        print(f"[INFO] Face crops already cached at {cache_dir}")
        return cache_dir

    print(f"[INFO] Pre-computing face crops...")
    print(f"[INFO] Dataset: {dataset_root}")
    print(f"[INFO] Cache: {cache_dir}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Compression: {compression}")

    if MTCNN is None:
        raise RuntimeError(
            "facenet-pytorch is required for face detection. Install with: pip install facenet-pytorch"
        )

    mtcnn = MTCNN(
        image_size=224,
        margin=20,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device,
        keep_all=False,
    )

    mtcnn.eval()

    manifest = {"crops": [], "compression": compression, "device": device}

    for label_name, label in [("real", 0), ("fake", 1)]:
        if label_name == "real":
            img_dir = os.path.join(
                dataset_root, "original_sequences", "youtube", compression, "images"
            )
        else:
            img_dir = os.path.join(
                dataset_root,
                "manipulated_sequences",
                "Deepfakes",
                compression,
                "images",
            )

        if not os.path.exists(img_dir):
            print(f"[WARN] Directory not found: {img_dir}")
            continue

        print(f"[INFO] Processing {label_name} images...")

        batch_images = []
        batch_info = []

        for video_id in os.listdir(img_dir):
            video_dir = os.path.join(img_dir, video_id)
            if not os.path.isdir(video_dir):
                continue

            for filename in os.listdir(video_dir):
                if not filename.endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(video_dir, filename)

                try:
                    image = Image.open(img_path).convert("RGB")
                    batch_images.append(image)
                    batch_info.append(
                        {
                            "path": img_path,
                            "label": label,
                            "filename": filename,
                            "video_id": video_id,
                        }
                    )
                except Exception as e:
                    print(f"[WARN] Failed to load {img_path}: {e}")
                    continue

                if len(batch_images) >= batch_size:
                    crops = process_batch(
                        batch_images, batch_info, mtcnn, cache_dir, manifest
                    )
                    batch_images = []
                    batch_info = []

        if batch_images:
            process_batch(batch_images, batch_info, mtcnn, cache_dir, manifest)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Pre-computation complete. {len(manifest['crops'])} crops cached.")
    return cache_dir


def process_batch(images, batch_info, mtcnn, cache_dir, manifest):
    """Process a batch of images through MTCNN."""
    crops = []
    for img, info in zip(images, batch_info):
        try:
            face = mtcnn(img)

            if face is not None:
                face_np = face.squeeze(0).permute(1, 2, 0).cpu().numpy()
                face_np = ((face_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
            else:
                img_resized = img.resize((224, 224))
                face_np = np.array(img_resized)

            crop_filename = f"crop_{len(manifest['crops']):06d}.npy"
            crop_path = os.path.join(cache_dir, crop_filename)
            np.save(crop_path, face_np)

            manifest["crops"].append(
                {
                    "path": crop_path,
                    "label": info["label"],
                    "original": info["path"],
                }
            )

        except Exception as e:
            print(f"[WARN] Failed to process {info['path']}: {e}")

    return crops


class FaceForensicsDataset(Dataset):
    """
    PyTorch Dataset for FaceForensics++ deepfake detection.

    Supports two modes:
    1. Pre-computed crops (recommended for training):
       - Use precompute_face_crops() first
       - Pass cache_dir pointing to cached crops
       - Much faster - no MTCNN in __getitem__

    2. On-the-fly detection (for exploration):
       - Uses MTCNN in __getitem__ (slow!)
       - Set use_precropped=False

    Binary classification:
    - Real = 0
    - Fake = 1

    Args:
        root (str): Root directory of the FaceForensics++ dataset.
        transform (transforms.Compose, optional): Transform to apply.
            Defaults to get_faceforensics_transforms().
        limit (int, optional): Maximum number of samples.
            Defaults to None (load all).
        compression (str): Compression level ('c23' or 'c40').
            Defaults to 'c23'.
        device (str): Device for MTCNN ('cuda' or 'cpu').
            Defaults to 'cuda' if available.
        use_precropped (bool): Use pre-computed crops if available.
            Defaults to True.
        cache_dir (str): Directory with pre-computed crops.
            Defaults to {root}/crops/.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        limit: int = None,
        compression: str = "c23",
        device: str = None,
        use_precropped: bool = True,
        cache_dir: str = None,
    ):
        self.root = root
        self.transform = transform or get_faceforensics_transforms(include_resize=False)
        self.limit = limit
        self.compression = compression
        self.use_precropped = use_precropped

        if cache_dir is None:
            cache_dir = os.path.join(root, "crops")
        self.cache_dir = cache_dir

        self.manifest_path = os.path.join(cache_dir, "manifest.json")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.mtcnn = None
        self.crops = []
        self.samples = []

        if use_precropped and os.path.exists(self.manifest_path):
            self._load_from_cache()
        else:
            self._setup_live_detection(device)

    def _load_from_cache(self):
        """Load samples from pre-computed crop cache."""
        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)

        self.crops = manifest["crops"]

        if self.limit:
            self.crops = self.crops[: self.limit]

        self.samples = [(crop["path"], crop["label"]) for crop in self.crops]

    def _setup_live_detection(self, device):
        """Set up for on-the-fly MTCNN detection (slow)."""
        if MTCNN is not None:
            self.mtcnn = MTCNN(
                image_size=224,
                margin=20,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device,
                keep_all=False,
            )
        else:
            print("[WARN] facenet-pytorch not installed. Using slow fallback mode.")

        self.samples = self._load_samples()

    def _load_samples(self):
        """Load all image paths from the dataset directory."""
        samples = []

        real_dir = os.path.join(
            self.root, "original_sequences", "youtube", self.compression, "images"
        )
        if os.path.exists(real_dir):
            samples.extend(self._collect_images_from_structure(real_dir, label=0))

        fake_dir = os.path.join(
            self.root, "manipulated_sequences", "Deepfakes", self.compression, "images"
        )
        if os.path.exists(fake_dir):
            samples.extend(self._collect_images_from_structure(fake_dir, label=1))

        if not samples:
            samples = self._load_flat_structure()

        if self.limit:
            samples = samples[: self.limit]

        return samples

    def _collect_images_from_structure(self, base_dir: str, label: int):
        """Recursively collect image paths from video folder structure."""
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
        """Load images from flat directory structure."""
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

        If using pre-computed crops (recommended):
            - Loads face crop from .npy file
            - Much faster than MTCNN

        If using live detection (slow):
            - Runs MTCNN on the fly
            - Falls back to resize if no face detected
        """
        image_path, label = self.samples[idx]

        if self.use_precropped and self.crops:
            face_crop = self._load_precropped(image_path)
        else:
            face_crop = self._detect_face_live(image_path)

        if self.transform:
            face_tensor = self.transform(face_crop)
        else:
            face_tensor = transforms.ToTensor()(face_crop)

        return face_tensor, label

    def _load_precropped(self, crop_path: str) -> Image.Image:
        """Load pre-computed face crop from numpy file."""
        try:
            crop_array = np.load(crop_path)
            return Image.fromarray(crop_array)
        except Exception:
            return Image.new("RGB", (224, 224))

    def _detect_face_live(self, image_path: str) -> Image.Image:
        """Detect face using MTCNN (slow)."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            if cv2 is not None:
                img_array = cv2.imread(image_path)
                if img_array is not None:
                    image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                else:
                    return Image.new("RGB", (224, 224))
            else:
                return Image.new("RGB", (224, 224))

        if self.mtcnn is not None:
            try:
                face = self.mtcnn(image)

                if face is not None:
                    face_np = face.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    face_np = ((face_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    return Image.fromarray(face_np)
                else:
                    return image.resize((224, 224), Image.BILINEAR)

            except Exception:
                return image.resize((224, 224), Image.BILINEAR)

        return image.resize((224, 224), Image.BILINEAR)
