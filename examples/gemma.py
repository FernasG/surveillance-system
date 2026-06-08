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
    
    def generate(self, messages: list[dict]):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        return self.processor.parse_response(response)
    

language_model = LanguageModel()
language_model.initialize()
messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://images.pexels.com/photos/821679/pexels-photo-821679.jpeg"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

response = language_model.generate(messages)

print(response)