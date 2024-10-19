"""
Implements text retrieval using CLIP embeddings and FAISS index.
"""

from typing import List, Dict
from src.modules.apple_clip import AppleCLIP
from src.modules.laion_clip import LaionCLIP
from src.repositories.load_faiss import ClipFaiss


class TextClipRetrieval:
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
        Initializes the ClipSearch class with the given CLIP models, FAISS index, and data.

        Args:
            top_k (int): The number of top results to retrieve.
            apple_clip (AppleCLIP): An instance of the AppleCLIP model.
            laion_clip (LaionCLIP): An instance of the LaionCLIP model.
            faiss (ClipFaiss): An instance of the ClipFaiss class for performing FAISS.
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
        Maps the search results (indices) to the corresponding video and frame information.

        Args:
            data (Dict): A dictionary mapping indices to video and frame information.
            indices (List[int]): A list of indices retrieved from the FAISS search.

        Returns:
            List: A list of mapped results containing video 
            and frame information for the given indices.
        """
        filtered_list = [data[indice] for indice in indices if indice in data]
        return filtered_list

    async def apple_text_retrieval(
        self,
        text: str
    ) -> List[Dict]:
        """
        Retrieves text data using the apple CLIP model.

        Args:
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        vector_embedding = await self._apple_clip.text_embedding(
            text=text
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

    async def laion_text_retrieval(
        self,
        text: str
    ) -> List[Dict]:
        """
        Retrieves text data using the laion CLIP model.

        Args:
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        vector_embedding = await self._laion_clip.text_embedding(
            text=text
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

    async def text_retrieval(
        self,
        model_type: str,
        text: str
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
            return await self.apple_text_retrieval(
                text=text
            )
        elif model_type == "laion_clip":
            return await self.laion_text_retrieval(
                text=text
            )
        else:
            return {
                "error": "Model type not supported"
            }
