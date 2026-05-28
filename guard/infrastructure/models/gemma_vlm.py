import torch
from PIL.Image import Image
from guard.core.entities import VLMResponse, VLMMessage
from guard.core.interfaces import VLMInterface
from transformers import AutoProcessor, AutoModelForMultimodalLM

class GemmaVLM(VLMInterface):
    def __init__(self):
        super().__init__()

        MODEL_ID = "google/gemma-4-E2B-it" # pass to .env

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    def generate(self, messages: list[VLMMessage], images: list[Image] = None) -> VLMResponse:
        hf_messages = [{"role": message.role, "content": message.content} for message in messages]
        
        inputs = self.processor(
            text=self.processor.apply_chat_template(hf_messages, add_generation_prompt=True),
            images=images,
            return_tensors="pt"
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)

        response_text = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        return VLMResponse(role="assistant", content=response_text.strip())

