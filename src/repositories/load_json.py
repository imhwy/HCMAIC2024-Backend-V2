"""
Loads and processes JSON data for video and frame mapping.
"""

import json


class LoadJson:
    """
    Loads a JSON file and creates a mapping of indices to video and frame IDs.
    """

    def __init__(
        self,
        json_url: str
    ) -> None:
        """
        Initializes the LoadJson class by loading the JSON data and creating a mapping.

        Args:
            json_url (str): The path to the JSON file containing the data.
        """
        with open(json_url, "r", encoding="utf-8") as f:
            self._mapping = json.load(f)
        self._data = {
            obj['indice']: {
                'video_id': obj['video_id'],
                'frame_id': obj['frame_id']
            } for obj in self._mapping
        }
