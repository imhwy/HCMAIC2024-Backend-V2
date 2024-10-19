"""
Implements text retrieval using CLIP embeddings and FAISS index.
"""

from io import BytesIO
from typing import List, Dict

from src.modules.apple_clip import AppleCLIP
from src.modules.laion_clip import LaionCLIP
from src.repositories.load_faiss import ClipFaiss


class ImageClipRetrieval:
    """
    Handles retrieval of text data using CLIP embeddings and FAISS index.
    """

    def __init__(
        self,
        top_k: int,
        apple_clip: AppleCLIP,
        laion_clip: LaionCLIP,
        faiss: ClipFaiss,
        data: Dict
    ) -> None:
        """
        Initializes the ClipSearch class with the provided CLIP models, FAISS index, and data.

        Args:
            top_k (int): The number of top search results to return.
            apple_clip (AppleCLIP): An instance of the AppleCLIP model for generating embeddings.
            laion_clip (LaionCLIP): An instance of the LaionCLIP model for generating embeddings.
            faiss (ClipFaiss): An instance of the ClipFaiss class for performing FAISS
            data (Dict): A dictionary mapping indices to video and frame information.
        """
        self._top_k = top_k
        self._apple_clip = apple_clip
        self._laion_clip = laion_clip
        self._faiss = faiss
        self._data = data

    async def mapping_results(
        self,
        data: Dict,
        indices: List[int]
    ) -> List:
        """
        Maps the search result indices to the corresponding data entries.

        Args:
            data (Dict): A dictionary where keys are indices and values are associated data.
            indices (List[int]): A list of indices retrieved from a search operation.

        Returns:
            List: A list of data entries corresponding to the indices.
        """
        filtered_list = [data[indice] for indice in indices if indice in data]
        return filtered_list

    async def apple_image_retrieval(
        self,
        image: BytesIO
    ) -> List[Dict]:
        """
        Retrieves text data using the apple CLIP model.

        Args:
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        vector_embedding = await self._apple_clip.image_embedding(
            image=image
        )
        indices = await self._faiss.apple_search(
            top_k=self._top_k,
            query_vectors=vector_embedding
        )
        result = await self.mapping_results(
            data=self._data,
            indices=indices[0]
        )
        return result

    async def laion_image_retrieval(
        self,
        image: BytesIO
    ) -> List[Dict]:
        """
        Retrieves text data using the laion CLIP model.

        Args:
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        vector_embedding = await self._laion_clip.image_embedding(
            image=image
        )
        indices = await self._faiss.laion_search(
            top_k=self._top_k,
            query_vectors=vector_embedding
        )
        result = await self.mapping_results(
            data=self._data,
            indices=indices[0]
        )
        return result

    async def image_retrieval(
        self,
        model_type: str,
        image: BytesIO
    ) -> List[Dict]:
        """
        Retrieves text data based on the specified model type.

        Args:
            model_type (str): The type of model to use for retrieval.
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        if model_type == "apple_clip":
            return await self.apple_image_retrieval(
                image=image
            )
        if model_type == "laion_clip":
            return await self.laion_image_retrieval(
                image=image
            )
        return {
            "error": "Model type not supported"
        }
