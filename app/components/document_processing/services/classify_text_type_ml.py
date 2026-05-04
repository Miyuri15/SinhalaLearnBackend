"""Optional MobileNetV2 text type classifier loader for printed vs handwritten classification."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Literal, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

TextTypeLabel = Literal["printed", "handwritten", "unknown"]
TextTypeResult = Tuple[TextTypeLabel, float]


class MobileNetV2TextTypeClassifier:
    """Lightweight wrapper around a trained MobileNetV2 .pth classifier."""

    def __init__(self, model_path: str, device: str | None = None):
        try:
            from torchvision import models, transforms
        except Exception as exc:
            raise RuntimeError("torchvision is required to load MobileNetV2 classifier") from exc

        self._models = models
        self._transforms = transforms
        self.model_path = str(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._build_model(self.model_path)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _build_model(self, model_path: str):
        model = self._models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 2)

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model.load_state_dict(checkpoint, strict=False)
        model.to(self.device)
        model.eval()
        return model

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> TextTypeResult:
        if image is None or image.size == 0:
            return "unknown", 0.0

        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        confidence, idx = torch.max(probs, dim=0)

        label: TextTypeLabel = "handwritten" if int(idx.item()) == 0 else "printed"
        return label, float(confidence.item())


class TextTypeClassifierLoader:
    """Singleton-style loader for MobileNet text type classifier."""

    _classifier: MobileNetV2TextTypeClassifier | None = None

    @classmethod
    def load(cls) -> MobileNetV2TextTypeClassifier | None:
        if cls._classifier is not None:
            return cls._classifier

        model_path = os.getenv("TEXT_TYPE_MOBILENET_PATH")

        if not model_path:
            logger.error("TEXT_TYPE_MOBILENET_PATH is not set")
            return None

        path = Path(model_path)
        if not path.exists():
            logger.error("Text type classifier path not found: %s", model_path)
            return None

        try:
            logger.info("Loading MobileNetV2 text type classifier...")

            cls._classifier = MobileNetV2TextTypeClassifier(str(path))

            logger.info("MobileNetV2 text type classifier loaded successfully.")
            return cls._classifier

        except Exception as exc:
            logger.error("Failed to load MobileNetV2 text type classifier: %s", exc)
            return None

    @classmethod
    def get(cls) -> MobileNetV2TextTypeClassifier | None:
        return cls._classifier