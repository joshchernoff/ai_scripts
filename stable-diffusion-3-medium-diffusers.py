import torch
from diffusers import StableDiffusion3Pipeline
from datetime import datetime
from huggingface_hub import login
import gc
import os

# Clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# Function to get user input with a default value
def get_user_input(prompt_text, default_value=None):
    if default_value:
        user_input = input(f"{prompt_text} (default: {default_value}): ")
    else:
        user_input = input(f"{prompt_text}: ")
    return user_input if user_input else default_value

# Function to save and print the image path
def save_and_print_image(image, output_dir="./output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"{output_dir}/stable-diffusion-3-medium-diffuser_{timestamp}.png"
    image.save(image_path)
    print(f"Image saved at: {image_path}")
    print(f"File path: file://{os.path.abspath(image_path)}")

# Get the Hugging Face token from the user or environment variable
hf_token = get_user_input("Enter your Hugging Face token", os.getenv('HUGGINGFACE_TOKEN'))

if not hf_token:
    raise ValueError("Hugging Face token not found. Please provide it as an input or set 'HUGGINGFACE_TOKEN' environment variable.")

# Login to Hugging Face
login(token=hf_token)

# Set environment variable for expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = "stabilityai/stable-diffusion-3-medium-diffusers"

# Load the pipeline without the T5 text encoder
pipe = StableDiffusion3Pipeline.from_pretrained(
    model, 
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16
)

pipe.to("cuda")

# Get user inputs with default values
prompt = get_user_input("Enter your prompt", "A beautiful landscape with mountains and a river at sunset")
negative_prompt = get_user_input("Enter your negative prompt", "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, .")
num_inference_steps = int(get_user_input("Enter the number of inference steps", "80"))
guidance_scale = float(get_user_input("Enter the guidance scale", "7.0"))

# Generate image using the pipeline
with torch.no_grad():
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

# Save and print the image path
save_and_print_image(image)

# Clear CUDA cache after execution
torch.cuda.empty_cache()
gc.collect()
