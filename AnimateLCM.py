import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from datetime import datetime
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

# Get the prompt, number of frames, and number of inference steps from the user
prompt = get_user_input("Enter your prompt", "A lighthouse in the ocean with waves crashing against it, 4k, high resolution")
negative_prompt = "bad quality, worse quality, low resolution"
num_frames = int(get_user_input("Enter the number of frames", "16"))
num_inference_steps = int(get_user_input("Enter the number of inference steps", "20"))

# Load adapter with float16 precision and move to GPU
adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16).to("cuda")
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

# Load LoRA weights and set adapters
pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
pipe.set_adapters(["lcm-lora"], [0.8])

# Enable VAE slicing and model CPU offload
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# Generate animation
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=num_frames,
    guidance_scale=7.0,
    num_inference_steps=num_inference_steps,
    generator=torch.Generator("cpu").manual_seed(0),
)

# Save and print the GIF path
frames = output.frames[0]
save_and_print_gif(frames)
