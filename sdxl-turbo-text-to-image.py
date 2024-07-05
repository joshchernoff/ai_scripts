from diffusers import AutoPipelineForText2Image
import torch
from datetime import datetime
import os

# Function to get user input with a default value
def get_user_input(prompt_text, default_value):
    user_input = input(f"{prompt_text} (default: {default_value}): ")
    return user_input if user_input else default_value

# Function to save and print the image path
def save_and_print_image(image, output_dir="./output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"{output_dir}/sdxl-turbo_{timestamp}.png"
    image.save(image_path)
    print(f"Image saved at: {image_path}")
    print(f"File path: file://{os.path.abspath(image_path)}")

# Get user inputs with default values
prompt = get_user_input("Enter your prompt", "Create a portrait of a grim dwarf warrior with a long-braided beard, iron armor, and a mighty axe. Depict them in a Tolkien-esque fantasy world")
negative_prompt = get_user_input("Enter your negative prompt", "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck")
num_inference_steps = int(get_user_input("Enter the number of inference steps", "2"))
strength = float(get_user_input("Enter the strength", "0.5"))
guidance_scale = float(get_user_input("Enter the guidance scale", "0.0"))

# Initialize the pipeline
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

# Generate the image
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    strength=strength,
    guidance_scale=guidance_scale
).images[0]

# Save and print the image path
save_and_print_image(image)