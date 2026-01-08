import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image


class ChestXRayClassification:
    """
    Chest X-ray image classification module for COVID-19 detection.

    This class:
    - Loads a pretrained DenseNet-121 model
    - Applies standardized image preprocessing
    - Performs inference on chest X-ray images
    - Outputs a predicted class label

    NOTE:
    This model is intended for research and decision-support only.
    It is NOT a substitute for professional medical diagnosis.
    """

    def __init__(self, model_path: str, device: torch.device = None):
        """
        Initialize the chest X-ray classifier.

        Args:
            model_path: File path to the trained model weights (.pth)
            device: Torch device to use (CPU or CUDA). Auto-detected if None.
        """
        # ------------------------------------------------------------------
        # Logging configuration
        # ------------------------------------------------------------------
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Class labels for prediction output
        self.class_names = ["covid19", "normal"]

        # Select device automatically if not provided
        self.device = (
            device
            if device
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.logger.info(f"Using device: {self.device}")

        # ------------------------------------------------------------------
        # Model initialization
        # ------------------------------------------------------------------
        self.model = self._build_model()
        self._load_model_weights(model_path)
        self.model.to(self.device)
        self.model.eval()  # Set model to inference mode

        # ------------------------------------------------------------------
        # Image preprocessing configuration
        # ------------------------------------------------------------------
        # ImageNet normalization parameters
        self.mean_nums = [0.485, 0.456, 0.406]
        self.std_nums = [0.229, 0.224, 0.225]

        # Transform pipeline applied to input images
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean_nums,
                std=self.std_nums
            )
        ])

    def _build_model(self):
        """
        Build the DenseNet-121 architecture with a custom classifier head.

        Returns:
            A DenseNet-121 model with a modified final layer
            matching the number of target classes.
        """
        model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(
            num_features,
            len(self.class_names)
        )
        return model

    def _load_model_weights(self, model_path: str):
        """
        Load pretrained model weights from disk.

        Args:
            model_path: Path to the saved model state dictionary

        Raises:
            Exception: If model weights fail to load
        """
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.logger.info(
                f"Model weights loaded successfully from {model_path}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load model weights: {str(e)}"
            )
            raise e

    def predict(self, img_path: str) -> str | None:
        """
        Perform inference on a single chest X-ray image.

        Args:
            img_path: File path to the input image

        Returns:
            Predicted class label ("covid19" or "normal"),
            or None if inference fails.
        """
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
            input_tensor = Variable(image_tensor).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                _, preds = torch.max(outputs, 1)
                class_index = preds.cpu().numpy()[0]
                predicted_class = self.class_names[class_index]

            self.logger.info(
                f"Chest X-ray prediction result: {predicted_class}"
            )

            return predicted_class

        except Exception as e:
            self.logger.error(
                f"Error during chest X-ray prediction: {str(e)}"
            )
            return None
