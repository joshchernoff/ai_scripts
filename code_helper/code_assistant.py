import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class CodeAssistant:
    def __init__(self, model_name="stabilityai/stable-code-instruct-3b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model.eval()
        self.model = self.model.cuda()

    def generate_response(self, user_content):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful and polite assistant",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        tokens = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.5,
            top_p=0.95,
            top_k=100,
            do_sample=True,
            use_cache=True
        )

        output = self.tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[-1]:], skip_special_tokens=False)[0]
        return output
