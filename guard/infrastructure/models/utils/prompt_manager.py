import yaml
from pathlib import Path
from loguru import logger

class PromptManager:
    def __init__(self, foldername: str = "../prompts"):
        self.folderpath = (Path(__file__).parent / foldername).resolve()
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        prompts = {}
        
        if not self.folderpath.exists():
            logger.warning(f"Prompt directory not found: {self.folderpath}")
            return prompts

        for file in self.folderpath.glob("*.yaml"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    key = file.stem
                    prompts[key] = yaml.safe_load(f)
                    logger.debug(f"Loaded prompt: {key}")
            except Exception as e:
                logger.error(f"Failed to load prompt file {file.name}: {e}")

        return prompts

    def build(self, prompt_name: str, **kwargs) -> str:
        if prompt_name not in self.prompts:
            logger.error(f"Prompt '{prompt_name}' not found. Available: {list(self.prompts.keys())}")
            raise KeyError(f"Prompt '{prompt_name}' not found.")
            
        template = self.prompts[prompt_name].get("template")

        if not template:
            raise ValueError(f"Prompt '{prompt_name}' is missing the 'template' key.")

        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Formatting error in prompt '{prompt_name}'. Missing variable or unescaped brace: {e}")
            raise e
        