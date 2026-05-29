import torch
from PIL.Image import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM

class LanguageModel:
    def __init__(self):
        self.MODEL_ID = "google/gemma-4-E2B-it"
        self.processor = None
        self.model = None

    def initialize(self):
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            torch_dtype=dtype
        )

        return None

    def generate(self, messages: list[dict], images: list[Image] = None):
        inputs = self.processor(
            text=self.processor.apply_chat_template(messages, add_generation_prompt=True),
            images=images,
            return_tensors="pt"
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)

        response_text = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        return response_text.strip()