import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class VideoQA:
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self) -> None:
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            dtype=torch.float16,
        ).to(self.device)

        self.model.eval()

    def answer(
            self,
            query: str,
            context: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided "
                    "video context. If the context does not contain enough "
                    "information, say that you cannot determine the answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Video context:\n{context}\n\n"
                    f"Question:\n{query}"
                ),
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        generated_tokens = output[
            0,
            inputs["input_ids"].shape[1]:,
        ]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )