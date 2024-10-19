"""
Implements a FAISS-based search for CLIP embeddings.
"""

from typing import List
import faiss
from torch import Tensor


class ClipFaiss:
    """
    Handles FAISS-based search operations for CLIP embeddings.
    """

    def __init__(
        self,
        apple_faiss_url: str,
        laion_faiss_url: str
    ) -> None:
        """
        Initializes the FAISS index and loads it onto a GPU.

        Args:
            faiss_url (str): The path to the FAISS index file.
            device_type (device): The device type (e.g., 'cpu' or 'cuda').

        """
        self._apple_index = faiss.read_index(apple_faiss_url)
        self._apple_res = faiss.StandardGpuResources()
        self._apple_gpu_index = faiss.index_cpu_to_gpu(
            provider=self._apple_res,
            device=1,
            index=self._apple_index
        )
        self._laion_index = faiss.read_index(laion_faiss_url)

    async def apple_search(
        self,
        top_k: int,
        query_vectors: Tensor
    ) -> List[int]:
        """
        Searches the Apple FAISS index for the top-k nearest neighbors.

        Args:
            top_k (int): The number of nearest neighbors to retrieve.
            query_vectors (Tensor): The query vectors to search against the index.

        Returns:
            List[int]: A list of indices of the top-k nearest neighbors.
        """
        query_vectors = query_vectors.cpu().detach().numpy()
        _, indices = self._apple_gpu_index.search(query_vectors, top_k)
        return indices

    async def laion_search(
        self,
        top_k: int,
        query_vectors: Tensor
    ) -> List[int]:
        """
        Searches the LAION FAISS index for the top-k nearest neighbors.

        Args:
            top_k (int): The number of nearest neighbors to retrieve.
            query_vectors (Tensor): The query vectors to search against the index.

        Returns:
            List[int]: A list of indices of the top-k nearest neighbors.
        """
        query_vectors = query_vectors.cpu().detach().numpy()
        _, indices = self._laion_index.search(query_vectors, top_k)
        return indices
