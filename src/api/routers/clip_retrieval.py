"""
This module defines a FastAPI router for handling clip text retrieval requests.
"""
import copy
import io
import time
from fastapi import (status,
                     Depends,
                     APIRouter,
                     HTTPException,
                     UploadFile,
                     File)

from src.api.schemas.clip import (RequestClipText,
                                  ResponseClip,
                                  ListResponseClip,
                                  MultiEventRequest,
                                  MultiModalResquest)
from src.services.service import Service
from src.api.dependencies.dependency import get_service
from src.utils.utility import count_non_empty_fields


clip_router = APIRouter(
    tags=["Clip"],
    prefix="/clip",
)


@clip_router.post(
    '/clipTextRetrieval',
    status_code=status.HTTP_200_OK,
    response_model=ListResponseClip
)
async def clip_text_retrieval(
    request: RequestClipText,
    service: Service = Depends(get_service)
) -> ListResponseClip:
    """
    Retrieves relevant text clips based on the provided query.

    Args:
        request (RequestClipText): The input data containing the text query and model type.
        service (Service): The service instance to handle the clip retrieval logic.

    Returns:
        ListResponseClipText: A list of text clips relevant to the input query.

    Raises:
        HTTPException: If the input text query is missing or an error occurs during processing.
    """
    if not request.text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is required"
        )
    try:
        a = time.time()
        result = await service.text_clip_retrieval.text_retrieval(
            model_type=request.model_type,
            text=request.text
        )
        print(time.time() - a)
        return ListResponseClip(
            data=[
                ResponseClip(**record) for record in result
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)) from e


@clip_router.post(
    "/searchByImage",
    status_code=status.HTTP_200_OK,
    response_model=ListResponseClip)
async def search_by_image(
    model_type: str,
    file: UploadFile = File(...),
    service: Service = Depends(get_service)
) -> ListResponseClip:
    """
    Perform a search using an uploaded image.

    Args:
        file (UploadFile): The image file to search with.
        service (Service): The service instance used for performing the search.

    Returns:
        ResponseResult: An object containing the search results.

    Raises:
        HTTPException: If no file is provided or if an error occurs during processing.
    """
    if not file.file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image is required"
        )
    try:
        a = time.time()
        contents = await file.read()
        image_stream = io.BytesIO(contents)
        result = await service.image_clip_retrieval.image_retrieval(
            model_type=model_type,
            image=image_stream
        )
        print(time.time() - a)
        return ListResponseClip(
            data=[
                ResponseClip(**record)for record in result
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e


@clip_router.post(
    "/multiEventSearch",
    status_code=status.HTTP_200_OK,
    response_model=ListResponseClip
)
async def multi_event_search(
    request: MultiEventRequest,
    service: Service = Depends(get_service)
) -> ListResponseClip:
    """
    """
    if not request.list_event:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="List of events is required"
        )
    try:
        a = time.time()
        result = await service.multi_event_retrieval.multi_event_search(
            model_type=request.model_type,
            list_event=request.list_event
        )
        print(time.time() - a)
        return ListResponseClip(
            data=[
                ResponseClip(**record) for record in result
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)) from e


@clip_router.post(
    "/multiModalSearch",
    status_code=status.HTTP_200_OK,
    response_model=ListResponseClip
)
async def multi_modal_search(
    request: MultiModalResquest,
    service: Service = Depends(get_service)
) -> ListResponseClip:
    """
    """
    if not request.list_ocr and not request.list_asr and not request.list_ocr:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="there is no features"
        )

    if count_non_empty_fields(request.text, request.list_ocr, request.list_asr) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="at least 2 in 3 fields are required"
        )

    try:
        list_ocr = [
            dict(
                obj, frame_id=f"{obj['frame_id']}.jpg"
            ) for obj in request.list_ocr
        ]
        list_asr = [
            dict(
                obj, frame_id=f"{obj['frame_id']}.jpg"
            ) for obj in request.list_asr
        ]

        if not request.text:
            result = await service.multi_event_retrieval.multi_event_search_with_non_text(
                list_ocr=list_ocr,
                list_asr=list_asr,
                priority=request.priority
            )
            return ListResponseClip(
                data=[
                    ResponseClip(**record) for record in result
                ]
            )

        if request.text:
            result = await service.multi_event_retrieval.multi_modal_search(
                model_type=request.model_type,
                text=request.text,
                list_ocr=request.list_ocr,
                list_asr=request.list_asr,
                priority=request.priority
            )
            return ListResponseClip(
                data=[
                    ResponseClip(**record) for record in result
                ]
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)) from e
