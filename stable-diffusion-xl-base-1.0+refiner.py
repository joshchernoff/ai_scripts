from diffusers import DiffusionPipeline
import torch
from datetime import datetime
import os

# Function to get user input for the prompt
def get_user_prompt():
    return input("Enter your prompt: ")

# Function to save and print the image path
def save_and_print_image(image, output_dir="./output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"{output_dir}/stable-diffusion-xl-base-1.0+refiner_{timestamp}.png"
    image.save(image_path)
    print(f"Image saved at: {image_path}")
    print(f"File path: file://{os.path.abspath(image_path)}")

# Get the prompt from the user
prompt = get_user_prompt()
negative_prompt = "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, ."

model = "stabilityai/stable-diffusion-xl-base-1.0"

# Load both base & refiner
base = DiffusionPipeline.from_pretrained(
    model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.enable_model_cpu_offload()

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 80
high_noise_frac = 0.5

# Run both experts
image = base(
    prompt=prompt,
    guidance_scale=7.0,
    negative_prompt=negative_prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=7.0,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

# Save and print the image path
save_and_print_image(image)
