"""
This module provides the inference service.
It imports the Service class from the src.services.service module and 
initializes an instance of it.
"""

from src.services.service import Service

service = Service()


async def get_service() -> Service:
    """
    Get the inference service instance.
    """
    return service
