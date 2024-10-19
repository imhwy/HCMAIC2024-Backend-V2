"""
"""

from typing import List, Dict, Union
from src.modules.apple_clip import AppleCLIP
from src.modules.laion_clip import LaionCLIP
from src.repositories.load_faiss import ClipFaiss


class MultiEventRetrieval:
    """
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
        """
        filtered_list = [data[indice] for indice in indices if indice in data]
        return filtered_list

    async def apple_text_retrieval(
        self,
        text: str
    ) -> List[Dict]:
        """
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

    async def find_common_elements_by_field(
        self,
        list_event: List[Dict],
        field: str = "video_id",
        frame_field: str = "frame_id"
    ) -> List[Dict]:
        """
        Tìm các đối tượng có cùng giá trị video_id trong list đầu tiên và xuất hiện trong n-1 list còn lại.
        Chỉ thêm vào danh sách kết quả nếu phần tử có mặt trong tất cả các danh sách còn lại.
        """
        base_list = list_event[0]  # Danh sách cơ sở (list đầu tiên)
        common_elements = []

        def extract_frame_number(frame_id: str) -> int:
            """Chuyển frame_id thành số nguyên để so sánh."""
            return int(frame_id.split('.')[0])

        for item in base_list:
            video_id_value = item[field]  # Lấy video_id từ phần tử hiện tại
            frame_id_value = extract_frame_number(
                item[frame_field])  # Lấy frame_id dưới dạng số nguyên

            # Kiểm tra xem phần tử hiện tại có trong tất cả các list khác không
            in_all_other_lists = all(
                any(
                    d[field] == video_id_value and extract_frame_number(
                        d[frame_field]) > frame_id_value
                    for d in lst
                )
                for lst in list_event[1:]
            )

            # Chỉ thêm phần tử vào danh sách kết quả nếu nó có mặt trong tất cả các danh sách khác
            if in_all_other_lists:
                common_elements.append(item)

        # Giới hạn kết quả trả về (nếu cần thiết)
        half_size = len(common_elements) // 100
        return common_elements[:half_size] if half_size > 0 else common_elements

    async def multi_event_search(
        self,
        model_type: str,
        list_event: List[str]
    ) -> List[Dict]:
        list_result = []
        for event in list_event:
            result = await self.text_retrieval(
                model_type=model_type,
                text=event
            )
            list_result.append(result)
        result = await self.find_common_elements_by_field(
            list_event=list_result,
            field="video_id"
        )
        return result

    async def prioritize_results(
        self,
        result: List[Dict],
        list_ocr: List[Dict],
        list_asr: List[Dict],
        priority: List[str]
    ) -> List[Dict]:
        """
        Prioritize results based on provided priority.
        """
        prioritized_results = []
        for item in priority:
            if item == 'asr':
                prioritized_results.extend(
                    [
                        r for r in result if r in list_asr
                    ]
                )
            elif item == 'ocr':
                prioritized_results.extend(
                    [
                        r for r in result if r in list_ocr
                    ]
                )
            elif item == 'clip':
                prioritized_results.extend(
                    [
                        r for r in result if r in result
                    ]
                )
        return prioritized_results

    async def multi_event_search_with_non_text(
        self,
        list_ocr: Union[List[Dict], None] = None,
        list_asr: Union[List[Dict], None] = None,
        priority: Union[List[str], None] = None
    ) -> List[Dict]:
        """
        """
        ocr_set = {tuple(item.items()) for item in list_ocr}
        asr_set = {tuple(item.items()) for item in list_asr}
        intersection = ocr_set.intersection(asr_set)
        result = [dict(item) for item in intersection]

        prioritized_results = await self.prioritize_results(
            result=result,
            list_ocr=list_ocr,
            list_asr=list_asr,
            priority=priority
        )
        return prioritized_results

    async def multi_modal_search(
        self,
        model_type: str = "apple_clip",
        text: str = None,
        list_ocr: Union[List[Dict], None] = None,
        list_asr: Union[List[Dict], None] = None,
        priority: Union[List[str], None] = None
    ) -> List[Dict]:
        """
        """
        combine = []
        result_clip = await self.text_retrieval(
            model_type=model_type,
            text=text
        )
        if list_asr and list_ocr:
            for item in priority:
                if item == 'asr':
                    combine.append(list_asr)
                elif item == 'ocr':
                    combine.append(list_ocr)
                elif item == 'clip':
                    combine.append(result_clip)
            result = await self.find_common_elements_by_field(
                list_event=combine,
                field="video_id"
            )
            return result
        elif list_asr:
            for item in priority:
                if item == 'asr':
                    combine.append(list_asr)
                elif item == 'clip':
                    combine.append(result_clip)
            print("pass cluster")
            result = await self.find_common_elements_by_field(
                list_event=combine,
                field="video_id"
            )
            print("pass cluster 2")
            return result
        elif list_ocr:
            for item in priority:
                if item == 'ocr':
                    combine.append(list_ocr)
                elif item == 'clip':
                    combine.append(result_clip)
            result = await self.find_common_elements_by_field(
                list_event=combine,
                field="video_id"
            )
            return result
        else:
            return []
