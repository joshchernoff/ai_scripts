from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-2-12b-chat')
model = AutoModelForCausalLM.from_pretrained(
    'stabilityai/stablelm-2-12b-chat',
    device_map="auto",
)

# system_prompt = """\
# You are a helpful assistant with access to the following functions. You must use them if required -\n
# [
#   {
#     "type": "function",
#     "function": {
#       "name": "TextToImage",
#       "description": "This function is able to create, draw, or illustrate an image from a text prompt.",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "prompt": {
#             "type": "string",
#             "description": "The description of image that the user wants to create."
#           }
#         },
#         "required": [
#           "prompt"
#         ]
#       }
#     }
#   }
# ]
# """
# messages = [
#     {'role': 'system', 'content': system_prompt},
#     {'role': "user", 'content': "Please, generate a picture of the Eiffel Tower at night!"}
# ]

# inputs = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors='pt'
# )

# tokens = model.generate(
#     inputs.to(model.device),
#     max_new_tokens=1024,
#     temperature=0.5,
#     do_sample=True
# )
# output = tokenizer.decode(tokens[:, inputs.shape[-1]:][0], skip_special_tokens=True)

# print(output)
# """
# [
#   {
#     "name": "TextToImage",
#     "arguments": {
#       "prompt": "Eiffel Tower at night."
#     }
#   }
# ]
# """


# prompt = [{'role': 'user', 'content': 'Implement snake game using pygame'}]
# inputs = tokenizer.apply_chat_template(
#     prompt,
#     add_generation_prompt=True,
#     return_tensors='pt'
# )

# tokens = model.generate(
#     inputs.to(model.device),
#     max_new_tokens=100,
#     temperature=0.7,
#     do_sample=True,
# )
# output = tokenizer.decode(tokens[:, inputs.shape[-1]:][0], skip_special_tokens=False)

# print(output)

