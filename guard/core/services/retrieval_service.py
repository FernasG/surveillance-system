import re
import cv2
import redis
from loguru import logger
from typing import Optional
from guard.infrastructure.models.utils.image_utils import cv2_to_base64
from guard.infrastructure.models.utils.prompt_manager import PromptManager
from guard.core.interfaces import VectorizerInterface, VectorStoreInterface, VLMInterface
from guard.core.exceptions import VLMGenerationError
from guard.core.entities import Settings, VLMMessage, SEARCH_PRIORITY_CHANNEL, SEARCH_GATE_KEY, SEARCH_GATE_TTL_S

class RetrievalService:
    def __init__(self, vectorizer: VectorizerInterface, store: VectorStoreInterface, vlm: VLMInterface, prompt_manager: PromptManager, redis_client: redis.Redis):
        self.settings = Settings()
        self.prompt_manager = prompt_manager
        self.vectorizer = vectorizer
        self.store = store
        self.vlm = vlm
        self.redis_client = redis_client

    def search_by_text(self, text: str, top_k: int = 5):
        log_context = {"text": text, "top_k": top_k}
        req_logger = logger.bind(queue_message=log_context)

        try:
            req_logger.info("Starting Text-Based RAG Query")

            query_vector = self.vectorizer.encode_text(text)
            search_result = self.store.search(query_vector, top_events=top_k)

            if not search_result:
                req_logger.info("No vectors found for the given query.")
                return {"results": []}

            extracted_data = []

            for idx, event_data in enumerate(search_result):
                metadata = event_data.get("metadata", {})
                elapsed_ms = metadata.get("elapsed_ms")
                video_path = metadata.get("video_path")

                if elapsed_ms is None or not video_path:
                    continue

                extracted_data.append({
                    "index": idx,
                    "uuid": event_data.get("frame_id"),
                    "event_id": event_data.get("event_id"),
                    "video_path": video_path,
                    "elapsed_ms": elapsed_ms,
                    "description": event_data.get("description", "Description unavailable")
                })

            if not extracted_data:
                return {"results": []}
            
            messages = self._setup_gemma_params(text, extracted_data)

            self._enter_search()
            try:
                response = self.vlm.generate(messages)
                eval_scores = self._parse_eval_scores(response.content)
            except VLMGenerationError as e:
                req_logger.warning(f"VLM re-ranking unavailable, falling back to vector-rank scores: {e}")
                eval_scores = {}
            finally:
                self._exit_search()

            final_results = []

            for frame_context in extracted_data:
                idx = frame_context["index"]
                
                if idx in eval_scores:
                    confidence_score = eval_scores[idx] / 10.0
                else:
                    req_logger.warning(f"VLM missing score for index {idx}. Applying fallback score.")
                    confidence_score = max(0.1, 0.5 - (idx * 0.05))

                frame_b64 = self._extract_frame_base64(
                    frame_context["video_path"],
                    frame_context["elapsed_ms"]
                )

                final_results.append({
                    "uuid": frame_context["uuid"],
                    "event_id": frame_context["event_id"],
                    "video_path": frame_context["video_path"],
                    "elapsed_ms": frame_context["elapsed_ms"],
                    "description": frame_context["description"],
                    "confidence_score": round(confidence_score, 3),
                    "frame_base64": frame_b64
                })

            final_results = sorted(final_results, key=lambda x: x["confidence_score"], reverse=True)

            return {"results": final_results}
        except Exception as e:
            req_logger.critical(f"Something went really wrong {e}")
            return {"error": str(e), "results": []}
        
    def _enter_search(self) -> None:
        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(SEARCH_GATE_KEY)
            pipe.expire(SEARCH_GATE_KEY, SEARCH_GATE_TTL_S)
            pipe.publish(SEARCH_PRIORITY_CHANNEL, "started")
            pipe.execute()
        except Exception as e:
            logger.warning(f"Failed to signal search start: {e}")

    def _exit_search(self) -> None:
        try:
            remaining = self.redis_client.decr(SEARCH_GATE_KEY)

            if remaining <= 0:
                self.redis_client.delete(SEARCH_GATE_KEY)

            self.redis_client.publish(SEARCH_PRIORITY_CHANNEL, "finished")
        except Exception as e:
            logger.warning(f"Failed to signal search finish: {e}")

    def _setup_gemma_params(self, text: str, extracted_data: list[dict]) -> list[VLMMessage]:
        formatted_descriptions = "\n".join([
            f"Index {data['index']}: {data['description']}"
            for data in extracted_data
        ])

        prompt_text = self.prompt_manager.build(
            prompt_name="search_evaluation_text",
            num_descriptions=len(extracted_data),
            formatted_descriptions=formatted_descriptions,
            user_query=text,
        )

        messages: list[VLMMessage] = [VLMMessage(role="user", content=prompt_text)]
        return messages

    def _parse_eval_scores(self, content: str) -> dict[int, float]:
        scores = {}

        for line in content.strip().splitlines():
            match = re.match(r"^\s*(\d+)\s*-\s*(\d+(?:\.\d+)?)\s*$", line)

            if not match:
                continue

            idx, score = match.groups()
            scores[int(idx)] = float(score)

        return scores

    def _extract_frame_base64(self, video_path: str, elapsed_ms: int) -> Optional[str]:
        cap = cv2.VideoCapture(video_path)

        try:
            if not cap.isOpened():
                logger.warning(f"Failed to open video file: {video_path}")
                return None

            cap.set(cv2.CAP_PROP_POS_MSEC, elapsed_ms)
            success, frame = cap.read()

            if success and frame is not None:
                return cv2_to_base64(frame)

            logger.warning(f"Failed to read frame at {elapsed_ms}ms from {video_path}")

            return None
        except Exception as e:
            logger.warning(f"Error extracting frame preview from {video_path}: {e}")
            return None
        finally:
            cap.release()