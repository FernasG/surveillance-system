import cv2
from loguru import logger
from guard.infrastructure.models.utils.image_utils import cv2_to_base64
from guard.infrastructure.models.utils.prompt_manager import PromptManager
from guard.core.interfaces import VectorizerInterface, VectorStoreInterface, VLMInterface
from guard.core.entities import Settings, VLMMessage

class RetrievalService:
    def __init__(self, vectorizer: VectorizerInterface, store: VectorStoreInterface, vlm: VLMInterface, prompt_manager: PromptManager):
        self.settings = Settings()
        self.prompt_manager = prompt_manager
        self.vectorizer = vectorizer
        self.store = store
        self.vlm = vlm

    def search_by_text(self, text: str, top_k: int = 5):
        log_context = { "text": text, "top_k": top_k }
        req_logger = logger.bind(queue_message=log_context)

        try:
            req_logger.info("Starting Text-Based RAG Query")

            query_vector = self.vectorizer.encode_text(text)
            search_result = self.store.search(query_vector, top_k=top_k)

            print(search_result)

            metadatas = search_result.get("metadatas", [])
            documents = search_result.get("documents", [])

            if not metadatas or not metadatas[0]:
                return {"results": []}

            extracted_data = []

            for idx, (metadata, doc_description) in enumerate(zip(metadatas[0], documents[0])):
                elapsed_ms = metadata.get("elapsed_ms")
                video_path = metadata.get("video_path")

                if not elapsed_ms or not video_path:
                    continue

                extracted_data.append({
                    "index": idx,
                    "video_path": video_path,
                    "elapsed_ms": elapsed_ms,
                    "description": doc_description
                })

            if not extracted_data:
                return {"results": []}
            
            messages = self._setup_gemma_params(text, extracted_data)

            response = self.vlm.generate(messages)

            final_results = []

            for eval_item in response.content:
                idx = eval_item.get("index")
                
                frame_context = next((item for item in extracted_data if item["index"] == idx), None)
                
                if frame_context is not None:
                    frame_b64 = None

                    try:
                        cap = cv2.VideoCapture(frame_context["video_path"])
                        cap.set(cv2.CAP_PROP_POS_MSEC, frame_context["elapsed_ms"])
                        success, frame = cap.read()
                        cap.release()
                        if success and frame is not None:
                            frame_b64 = cv2_to_base64(frame)
                    except Exception as e:
                        logger.warning(f"Could not extract final frame preview: {e}")

                    final_results.append({
                        "video_path": frame_context["video_path"],
                        "elapsed_ms": frame_context["elapsed_ms"],
                        "confidence_score": eval_item.get("confidence_score", 0.0),
                        "description": frame_context["description"],
                        "frame_base64": frame_b64
                    })

            final_results = sorted(final_results, key=lambda x: x["confidence_score"], reverse=True)

            return {"results": final_results}
        except Exception as e:
            req_logger.critical(f"Something went really wrong {e}")
            return {"error": str(e), "results": []}
        
    def _setup_gemma_params(self, text: str, extracted_data: list[dict]) -> list[VLMMessage]:
        formatted_descriptions = "\n".join([
            f"Index {data['index']}: {data['description']}"
            for data in extracted_data
        ])

        prompt_text = (
            f"You are a video surveillance AI assistant. I have provided {len(extracted_data)} distinct sequential frame descriptions extracted from a security camera feed.\n"
            f"Carefully analyze ALL provided descriptions and evaluate how well each one matches this user search query: '{text}'\n\n"
            f"Available Descriptions:\n"
            f"{formatted_descriptions}\n\n"
            f"For each item, provide a confidence score from 0.0 to 1.0. Lower your score if the description does not show a clear subject or is otherwise irrelevant.\n"
            f"Respond strictly in this JSON format:\n"
            f"[\n"
            f"  {{\n"
            f"    \"index\": 0,\n"
            f"    \"confidence_score\": 0.0\n"
            f"  }}\n"
            f"]"
        )

        messages: list[VLMMessage] = [
            VLMMessage(role="user", content=prompt_text)
        ]

        return messages