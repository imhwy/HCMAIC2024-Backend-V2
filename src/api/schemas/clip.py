"""
Schemas for clip text retrieval API.
"""

from typing import (List,
                    Dict)
from pydantic import BaseModel


class RequestClipText(BaseModel):
    """
    Request schema for clip text retrieval.
    """
    model_type: str
    text: str


class ResponseClip(BaseModel):
    """
    Response schema for individual text clip.
    """
    frame_id: str
    video_id: str


class ListResponseClip(BaseModel):
    """
    Response schema for a list of text clips.
    """
    data: List[ResponseClip]

class MultiEventRequest(BaseModel):
    """
    """
    model_type: str
    list_event: List[str]

class MultiModalResquest(BaseModel):
    """
    """
    model_type: str
    text: str
    list_ocr: List[Dict]
    list_asr: List[Dict]
    priority: List[str]
