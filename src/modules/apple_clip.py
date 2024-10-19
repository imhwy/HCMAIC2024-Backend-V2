"""
This module is used for Apple CLIP model-based text and image embedding.
"""

import torch
from torch import device, Tensor
import torch.nn.functional as F
from PIL import Image
from open_clip.factory import (create_model,
                               image_transform_v2,
                               get_tokenizer)


class AppleCLIP:
    """
    A class for handling text and image embeddings using a CLIP model.
    Provides methods to generate embeddings for text and images.
    """

    def __init__(
        self,
        model: create_model,
        processor: image_transform_v2,
        tokenizer: get_tokenizer,
        device_type: device
    ) -> None:
        """
        Initialize the AppleCLIP class.

        Args:
            model (create_model): The CLIP model to be used for encoding.
            processor (image_transform_v2): The image transformation function.
            tokenizer (get_tokenizer): The tokenizer for processing text inputs.
            device_type (device): The device on which the model will run (e.g., CPU or GPU).
        """
        self._model = model
        self._processor = processor
        self._tokenizer = tokenizer
        self._device_type = device_type

    async def text_embedding(
        self,
        text: str
    ) -> Tensor:
        """
        Generate a text embedding using the CLIP model.

        Args:
            text (str): The input text to be encoded.

        Returns:
            Tensor: The normalized text embedding as a PyTorch tensor.
        """
        text = self._tokenizer(
            text,
            context_length=self._model.context_length
        ).to(self._device_type)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self._model.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    async def image_embedding(
        self,
        image
    ) -> Tensor:
        """
        Generate an image embedding using the CLIP model.

        Args:
            image: The input image file (path or file-like object) to be encoded.

        Returns:
            Tensor: The normalized image embedding as a PyTorch tensor.
        """
        image = Image.open(image).convert("RGB")
        image = self._processor(image).unsqueeze(0).to(self._device_type)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self._model.encode_image(image)
            image_features = F.normalize(image_features, dim=-1)
        return image_features
