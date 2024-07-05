from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-2-1_6b")
model = AutoModelForCausalLM.from_pretrained(
  "stabilityai/stablelm-2-1_6b",
  torch_dtype="auto",
)
model.cuda()
inputs = tokenizer("The weather is always wonderful", return_tensors="pt").to(model.device)
tokens = model.generate(
  **inputs,
  max_new_tokens=64,
  temperature=0.70,
  top_p=0.95,
  do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
