"""Universal YOLO detector supporting v8 and v12 for watermark detection."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
from enum import Enum

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class YOLOVersion(Enum):
    """YOLO version selection."""

    V8 = "v8"
    V12 = "v12"
    V11 = "v11"


class YOLOWatermarkDetector:
    """YOLOv8-based watermark detector optimized for edge deployment."""

    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = "cpu",
        use_onnx: bool = True,
        verbose: bool = False,
        version: YOLOVersion = YOLOVersion.V8,
    ):
        """Initialize YOLO watermark detector supporting v8 and v12.

        Args:
            model_path: Path to model (.pt or .onnx)
            conf_thres: Confidence threshold (0-1)
            iou_thres: NMS IoU threshold (0-1)
            device: Inference device ('cpu', 'cuda', 'auto')
            use_onnx: Use ONNX Runtime if available
            verbose: Enable verbose logging
            version: YOLOVersion.V8, YOLOVersion.V12, or YOLOVersion.V11 (default: V8)
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics>=8.3.0"
            )

        self.version = version
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.verbose = verbose
        self.device = device
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.model = None
        self.session = None

        # Model cache directory
        self.model_cache = Path.home() / ".pdf_watermark" / "models"
        self.model_cache.mkdir(parents=True, exist_ok=True)

        # Auto-resolve model path based on version if needed
        if model_path == "yolov8n-seg.pt":
            if version == YOLOVersion.V12:
                model_path = "yolov12n-seg.pt"
                if self.verbose:
                    print(
                        f"[YOLO] Auto-switched to {version.value} model: {model_path}"
                    )
            elif version == YOLOVersion.V11:
                model_path = "yolo11x-watermark.pt"
                if self.verbose:
                    print(
                        f"[YOLO] Auto-switched to {version.value} model: {model_path}"
                    )

        # Load model
        model_path = str(model_path)
        if not Path(model_path).exists():
            # Try to load from cache or download
            cached_model = self.model_cache / Path(model_path).name
            if cached_model.exists():
                model_path = str(cached_model)
            else:
                # YOLO will auto-download
                pass

        # Load model
        if self.use_onnx and model_path.endswith(".onnx"):
            self._load_onnx_model(model_path)
        else:
            self._load_torch_model(model_path)

    def _load_torch_model(self, model_path: str) -> None:
        """Load PyTorch model."""
        try:
            self.model = YOLO(model_path)
            self.model.fuse()  # Model fusion for acceleration
            if self.verbose:
                print(
                    f"[YOLO{self.version.value.upper()}] Loaded PyTorch model: {model_path}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def _load_onnx_model(self, model_path: str) -> None:
        """Load ONNX Runtime model."""
        if not ONNX_AVAILABLE:
            if self.verbose:
                print(
                    f"[YOLO{self.version.value.upper()}] ONNX not available, falling back to PyTorch"
                )
            self._load_torch_model(model_path.replace(".onnx", ".pt"))
            return

        try:
            providers = ["CPUExecutionProvider"]
            if self.device in ("cuda", "auto"):
                providers.insert(0, "CUDAExecutionProvider")

            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]

            if self.verbose:
                print(
                    f"[YOLO{self.version.value.upper()}] Loaded ONNX model: {model_path}"
                )
                print(
                    f"[YOLO{self.version.value.upper()}] Providers: {self.session.get_providers()}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def preprocess(
        self, image_rgb: np.ndarray, target_size: int = 640
    ) -> Tuple[np.ndarray, float, int, int]:
        """Preprocess image for YOLOv8.

        Args:
            image_rgb: Input image in RGB format
            target_size: Target size for inference (default 640)

        Returns:
            Tuple of (preprocessed_tensor, scale, pad_w, pad_h)
        """
        h, w = image_rgb.shape[:2]

        # Resize maintaining aspect ratio
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        img_resized = cv2.resize(
            image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        # Pad to target size
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        img_padded = cv2.copyMakeBorder(
            img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Normalize and transpose
        img_tensor = img_padded.astype(np.float32) / 255.0
        img_tensor = img_tensor.transpose(2, 0, 1)  # HWC -> CHW
        img_tensor = img_tensor[np.newaxis, ...]  # Add batch dimension

        return img_tensor, scale, pad_w, pad_h

    def postprocess(
        self,
        outputs,
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_shape: Tuple[int, int, int],
        target_size: int = 640,
    ) -> np.ndarray:
        """Postprocess YOLOv8 outputs to watermark mask.

        Args:
            outputs: Model outputs
            scale: Preprocessing scale factor
            pad_w: Padding width
            pad_h: Padding height
            orig_shape: Original image shape (H, W, C)
            target_size: Target size used in preprocessing

        Returns:
            Binary watermark mask
        """
        # Handle both PyTorch and ONNX outputs
        if isinstance(outputs, list):
            # ONNX output format
            proto_masks = outputs[1] if len(outputs) > 1 else None
        else:
            # PyTorch Results object
            if hasattr(outputs, "masks") and outputs.masks is not None:
                masks_data = outputs.masks.data.cpu().numpy()
            else:
                return np.zeros(orig_shape[:2], dtype=np.uint8)
            # Use masks data directly
            proto_masks = masks_data

        if proto_masks is None or len(proto_masks) == 0:
            return np.zeros(orig_shape[:2], dtype=np.uint8)

        # Combine all detected masks
        if isinstance(proto_masks, np.ndarray):
            if proto_masks.ndim == 3:
                # Multiple masks
                combined_mask = np.max(proto_masks, axis=0)
            else:
                # Single mask
                combined_mask = proto_masks
        else:
            combined_mask = np.zeros((target_size, target_size), dtype=np.float32)

        # Remove padding
        if pad_h > 0:
            combined_mask = combined_mask[: target_size - pad_h, :]
        if pad_w > 0:
            combined_mask = combined_mask[:, : target_size - pad_w]

        # Resize to original dimensions
        combined_mask = cv2.resize(
            combined_mask,
            (orig_shape[1], orig_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Binarize
        combined_mask = (combined_mask > 0.5).astype(np.uint8) * 255

        return combined_mask

    def detect_watermark_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Detect watermark mask using YOLO.

        Args:
            image_rgb: Input image in RGB format

        Returns:
            Binary watermark mask (0/255)
        """
        if self.verbose:
            print(f"[YOLO{self.version.value.upper()}] Detecting watermark...")

        # Preprocess
        input_tensor, scale, pad_w, pad_h = self.preprocess(image_rgb)

        # Inference
        if self.session is not None:
            # ONNX mode
            outputs = self.session.run(
                self.output_names, {self.input_name: input_tensor}
            )
        else:
            # PyTorch mode
            results = self.model.predict(
                image_rgb,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=640,
                device=self.device if self.device != "auto" else None,
                verbose=self.verbose,
                half=self.device == "cuda",
            )
            outputs = results[0] if isinstance(results, list) else results

        # Postprocess
        mask = self.postprocess(outputs, scale, pad_w, pad_h, image_rgb.shape)

        if self.verbose:
            coverage = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1]) * 100
            print(
                f"[YOLO{self.version.value.upper()}] Watermark coverage: {coverage:.2f}%"
            )

        return mask

    def refine_mask(self, mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Refine mask with morphological operations.

        Args:
            mask: Input watermark mask
            kernel_size: Morphological kernel size

        Returns:
            Refined mask
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return refined


# Backward compatibility alias
YOLOv8WatermarkDetector = YOLOWatermarkDetector
