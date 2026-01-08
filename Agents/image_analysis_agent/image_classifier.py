import os
import json
import base64
from mimetypes import guess_type
from typing import TypedDict

from langchain_core.output_parsers import JsonOutputParser


class ClassificationDecision(TypedDict):
    """
    Structured output schema for image classification decisions.

    Fields:
        image_type: Classified image category (e.g., CHEST X-RAY, OTHER, NON-MEDICAL)
        reasoning: Explanation of how the classification decision was made
        confidence: Confidence score between 0.0 and 1.0
    """
    image_type: str
    reasoning: str
    confidence: float


class ImageClassifier:
    """
    Image classification utility using a vision-capable LLM (e.g., GPT-4o Vision).

    This class:
    - Converts local images into base64-encoded data URLs
    - Sends images and instructions to a vision model
    - Parses structured JSON output describing the image type

    Intended for routing medical images to appropriate downstream agents.
    """

    def __init__(self, vision_model):
        """
        Initialize the image classifier.

        Args:
            vision_model: Vision-capable LLM instance used for image analysis
        """
        self.vision_model = vision_model

        # JSON parser enforcing structured classification output
        self.json_parser = JsonOutputParser(
            pydantic_object=ClassificationDecision
        )

    def local_image_to_data_url(self, image_path: str) -> str:
        """
        Convert a local image file into a base64-encoded data URL.

        This format is required by vision-capable LLM APIs that accept
        inline image inputs.

        Args:
            image_path: Path to the local image file

        Returns:
            A data URL string containing the base64-encoded image
        """
        mime_type, _ = guess_type(image_path)

        # Fallback MIME type if detection fails
        if mime_type is None:
            mime_type = "application/octet-stream"

        # Read and encode image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return f"data:{mime_type};base64,{base64_encoded_data}"

    def classify_image(self, image_path: str) -> dict:
        """
        Analyze an image and classify whether it is medical and its type.

        Classification categories:
        - CHEST X-RAY
        - OTHER (medical but not chest X-ray)
        - NON-MEDICAL

        Args:
            image_path: Path to the image to be analyzed

        Returns:
            A dictionary containing:
            - image_type
            - reasoning
            - confidence
        """
        print(f"[ImageAnalyzer] Analyzing image: {image_path}")

        # Vision prompt with explicit JSON output constraints
        vision_prompt = [
            {
                "role": "system",
                "content": "You are an expert in medical imaging. Analyze the uploaded image."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            """
                            Determine if this is a medical image.
                            If it is medical, classify it as:
                            - 'CHEST X-RAY'
                            - 'OTHER'
                            If it is not medical, return:
                            - 'NON-MEDICAL'

                            Respond ONLY in valid JSON using the following structure:
                            {
                              "image_type": "IMAGE TYPE",
                              "reasoning": "Clear explanation of the classification",
                              "confidence": 0.95
                            }
                            """
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.local_image_to_data_url(image_path)
                        }
                    }
                ]
            }
        ]

        # Invoke the vision model
        response = self.vision_model.invoke(vision_prompt)

        try:
            # Parse and validate structured JSON output
            response_json = self.json_parser.parse(response.content)
            return response_json

        except json.JSONDecodeError:
            # Defensive fallback in case of malformed LLM output
            print("[ImageAnalyzer] Warning: Model response was not valid JSON.")
            return {
                "image_type": "unknown",
                "reasoning": "Invalid JSON response from vision model",
                "confidence": 0.0
            }
