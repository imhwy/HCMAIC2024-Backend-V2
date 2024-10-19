"""
Service class for initializing and managing the CLIP retrieval system.
"""

# import os
from dotenv import load_dotenv
import torch
from open_clip import (create_model_from_pretrained,
                       get_tokenizer)
from transformers import (CLIPProcessor,
                          AutoTokenizer,
                          CLIPModel)

# from src.utils.utility import convert_value
from src.modules.apple_clip import AppleCLIP
from src.modules.laion_clip import LaionCLIP
from src.repositories.load_faiss import ClipFaiss
from src.repositories.load_json import LoadJson
from src.services.text_clip_retrieval import TextClipRetrieval
from src.services.image_clip_retrieval import ImageClipRetrieval
from src.services.multi_event_retrieval import MultiEventRetrieval

load_dotenv()

APPLE_CLIP_MODEL = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
APPLE_CLIP_TOKENIZER = "ViT-H-14"
LAION_CLIP_MODEL = "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
LAION_CLIP_TOKENIZER = "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
APPLE_FAISS = "/kaggle/input/apple-clip/apple.faiss"
LAION_FAISS = "/kaggle/input/laion-clip/laion.faiss"
JSON_CLIP = "/kaggle/input/json-clip/clip.json"
TOP_K = 1500


class Service:
    """
    Initializes and provides access to the CLIP retrieval service.
    """

    def __init__(
        self,
        apple_clip_model=APPLE_CLIP_MODEL,
        apple_clip_tokenizer=APPLE_CLIP_TOKENIZER,
        laion_clip_model=LAION_CLIP_MODEL,
        laion_clip_tokenizer=LAION_CLIP_TOKENIZER,
        apple_clip_faiss=APPLE_FAISS,
        laion_clip_faiss=LAION_FAISS,
        json_clip=JSON_CLIP,
        top_k=TOP_K
    ) -> None:
        """
        Sets up the necessary components for the CLIP retrieval service.

        Args:
            top_k (int): The number of top results to return during retrieval.
        """
        self._data = LoadJson(
            json_url=json_clip
        )._data
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._apple_model, self._apple_processor = create_model_from_pretrained(
            apple_clip_model
        )
        self._apple_model.to(self._device)
        self._apple_tokenizer = get_tokenizer(apple_clip_tokenizer)
        self._laion_model, self._laion_processor = create_model_from_pretrained(
            laion_clip_model
        )
        self._laion_model.to(self._device)
        self._laion_tokenizer = get_tokenizer(laion_clip_tokenizer)
        self._apple_clip = AppleCLIP(
            model=self._apple_model,
            processor=self._apple_processor,
            tokenizer=self._apple_tokenizer,
            device_type=self._device
        )
        self._laion_clip = LaionCLIP(
            model=self._laion_model,
            processor=self._laion_processor,
            tokenizer=self._laion_tokenizer,
            device_type=self._device
        )
        self._faiss = ClipFaiss(
            apple_faiss_url=apple_clip_faiss,
            laion_faiss_url=laion_clip_faiss
        )
        self._text_clip_retrieval = TextClipRetrieval(
            top_k=top_k,
            apple_clip=self._apple_clip,
            laion_clip=self._laion_clip,
            faiss=self._faiss,
            data=self._data
        )
        self._image_clip_retrieval = ImageClipRetrieval(
            top_k=top_k,
            apple_clip=self._apple_clip,
            laion_clip=self._laion_clip,
            faiss=self._faiss,
            data=self._data
        )
        self._multi_event_retrieval = MultiEventRetrieval(
            top_k=top_k,
            apple_clip=self._apple_clip,
            laion_clip=self._laion_clip,
            faiss=self._faiss,
            data=self._data
        )

    @property
    def text_clip_retrieval(self):
        """
        Provides access to the initialized CLIP retrieval service.

        Returns:
            ClipRetrieval: The CLIP retrieval service instance.
        """
        return self._text_clip_retrieval

    @property
    def image_clip_retrieval(self):
        """
        Provides access to the initialized CLIP retrieval service.

        Returns:
            ClipRetrieval: The CLIP retrieval service instance.
        """
        return self._image_clip_retrieval

    @property
    def multi_event_retrieval(self):
        """
        Provides access to the initialized CLIP retrieval service.

        Returns:
            ClipRetrieval: The CLIP retrieval service instance.
        """
        return self._multi_event_retrieval
