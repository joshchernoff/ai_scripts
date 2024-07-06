import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_gif
from datetime import datetime
from PIL import Image
import os

# Function to get user input with a default value
def get_user_input(prompt_text, default_value):
    user_input = input(f"{prompt_text} (default: {default_value}): ")
    return user_input if user_input else default_value

# Function to save and print the GIF path
def save_and_print_gif(frames, output_dir="./output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = f"{output_dir}/animatelcm_{timestamp}.gif"
    export_to_gif(frames, gif_path)
    print(f"GIF saved at: {gif_path}")
    print(f"File path: file://{os.path.abspath(gif_path)}")

# Get the number of frames and number of inference steps from the user
num_frames = int(get_user_input("Enter the number of frames", "16"))
num_inference_steps = int(get_user_input("Enter the number of inference steps", "20"))
image_path = get_user_input("Enter the path to the input image", "./input_image.jpg")

# Load the input image
init_image = Image.open(image_path).convert("RGB")

# Load the pipeline with float16 precision and move to GPU
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16).to("cuda")

# Generate animation
output = pipeline(
    image=init_image,
    num_frames=num_frames,
    num_inference_steps=num_inference_steps,
    guidance_scale=7.0,
    generator=torch.Generator("cpu").manual_seed(0),
)

# Save and print the GIF path
frames = output.frames
save_and_print_gif(frames)
