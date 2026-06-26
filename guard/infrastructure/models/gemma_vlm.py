import requests
from loguru import logger
from typing import Literal
from PIL.Image import Image
from guard.core.entities import Settings
from guard.core.entities import VLMResponse, VLMMessage
from guard.core.interfaces import VLMInterface
from .utils.image_utils import pil_to_base64
from .utils.json_parser import parse_json_markdown

class GemmaVLM(VLMInterface):
    def __init__(self):
        super().__init__()

        settings = Settings()

        self.api_url = settings.ollama_api_url
        self.model_name = settings.ollama_model_name
        self.request_timeout = 300
        self.model_temperature = 0.1

    def generate(self, messages: list[VLMMessage], images: list[Image] = None, format_response: Literal["json"] = None) -> VLMResponse:
        ollama_messages = []

        for message in messages:
            if message.role != "user" or not images:
                ollama_messages.append({
                    "role": message.role,
                    "content": message.content
                })

                continue

            content_list = [{"type": "text", "text": message.content}]
            
            for img in images:
                b64_str = pil_to_base64(img)
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_str}"
                    }
                })
            
            ollama_messages.append({
                "role": "user",
                "content": content_list
            })

        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "temperature": self.model_temperature,
            "format": format_response
        }

        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=self.request_timeout
            )
            response.raise_for_status()
            response_json = response.json()

            content = response_json["choices"][0]["message"]["content"].strip()
            
            if format_response:
                content = parse_json_markdown(content)

            return VLMResponse(role="assistant", content=content)
        except requests.exceptions.RequestException as e:
            message = f"Error communicating with Ollama server: {str(e)}"

            logger.error(message)

            return VLMResponse(role="assistant", content=message)
