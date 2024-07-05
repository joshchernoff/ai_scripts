import torch
from diffusers import StableDiffusion3Pipeline
from datetime import datetime
from huggingface_hub import login
import gc
import os

# Clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# Login to Hugging Face
login(token="")

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

# Set variables
prompt = "(portrait), (donald trump), A dingy dirty jail cell with iron bars, a small cot, and a single, flickering overhead light. A prisoner (donald trump) sits on the cot, head in hands, with a guard standing watch nearby. fantisy, Political Art"
negative_prompt = "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, ."

num_inference_steps = 80  # Reduce the number of steps
guidance_scale = 7.0

# Generate image using the pipeline
with torch.no_grad():
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

# Save image
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image.save(f"./output/stable-diffusion-3-medium-diffuser{timestamp}.png")

# Clear CUDA cache after execution
torch.cuda.empty_cache()
gc.collect()


#####################################
####### WORKS
#####################################

# import torch
# from diffusers import StableDiffusion3Pipeline
# from huggingface_hub import login
# import gc
# import os

# # Clear CUDA cache
# torch.cuda.empty_cache()
# gc.collect()

# # Login to Hugging Face
# login(token="")

# # Set environment variable for expandable segments
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # Load the pipeline without the T5 text encoder
# pipe = StableDiffusion3Pipeline.from_pretrained(
#     "stabilityai/stable-diffusion-3-medium-diffusers", 
#     text_encoder_3=None,
#     tokenizer_3=None,
#     torch_dtype=torch.float16
# )
# pipe.to("cuda")

# # Set variables
# prompt = "A cat holding a sign that says hello world"
# negative_prompt = "4 paws"
# num_inference_steps = 40  # Reduce the number of steps
# guidance_scale = 7.0

# # Generate image using the pipeline
# with torch.no_grad():
#     image = pipe(
#         prompt,
#         negative_prompt=negative_prompt,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale,
#     ).images[0]

# # Save image
# image.save("stable-diffusion-3-medium-diffusers.png")

# # Clear CUDA cache after execution
# torch.cuda.empty_cache()
# gc.collect()

