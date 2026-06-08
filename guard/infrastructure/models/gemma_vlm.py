from PIL.Image import Image
from guard.core.entities import VLMResponse, VLMMessage
from guard.core.interfaces import VLMInterface

class GemmaVLM(VLMInterface):
    def __init__(self):
        super().__init__()

    def generate(self, messages: list[VLMMessage], images: list[Image] = None) -> VLMResponse:
        return

