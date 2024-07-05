import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
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
    image_path = f"{output_dir}/stable-cascade_{timestamp}.png"
    image.save(image_path)
    print(f"Image saved at: {image_path}")
    print(f"File path: file://{os.path.abspath(image_path)}")

# Get user inputs with default values
prompt = get_user_input("Enter your prompt", "Create a portrait of a grim dwarf warrior with a long-braided beard, iron armor, and a mighty axe. Depict them in a Tolkien-esque fantasy world")
negative_prompt = get_user_input("Enter your negative prompt", "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck")
height = int(get_user_input("Enter the image height", "1024"))
width = int(get_user_input("Enter the image width", "1024"))
guidance_scale_prior = float(get_user_input("Enter the guidance scale for prior", "4.0"))
guidance_scale_decoder = float(get_user_input("Enter the guidance scale for decoder", "0.0"))
num_images_per_prompt = int(get_user_input("Enter the number of images per prompt", "1"))
num_inference_steps_prior = int(get_user_input("Enter the number of inference steps for prior", "20"))
num_inference_steps_decoder = int(get_user_input("Enter the number of inference steps for decoder", "10"))

# Load the models
prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)

# Generate image embeddings
prior.enable_model_cpu_offload()
prior_output = prior(
    prompt=prompt,
    height=height,
    width=width,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale_prior,
    num_images_per_prompt=num_images_per_prompt,
    num_inference_steps=num_inference_steps_prior
)

# Decode the image
decoder.enable_model_cpu_offload()
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings.to(torch.float16),
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale_decoder,
    output_type="pil",
    num_inference_steps=num_inference_steps_decoder
).images[0]

# Save and print the image path
save_and_print_image(decoder_output)
