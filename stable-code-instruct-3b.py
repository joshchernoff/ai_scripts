import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(user_prompt):
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-code-instruct-3b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stable-code-instruct-3b", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    model = model.cuda()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and polite assistant",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    tokens = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.5,
        top_p=0.95,
        top_k=100,
        do_sample=True,
        use_cache=True
    )

    output = tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[-1]:], skip_special_tokens=False)[0]
    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a specified prompt")
    parser.add_argument("prompt", type=str, help="The prompt to generate text from")
    args = parser.parse_args()
    main(args.prompt)
