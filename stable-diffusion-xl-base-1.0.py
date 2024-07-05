from datetime import datetime
from diffusers import StableDiffusionXLPipeline
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

model = "stabilityai/stable-diffusion-xl-base-1.0"

n_steps = 80
high_noise_frac = 0.7
guidance_scale = 7.0

prompt = "(portrait), (donald trump), A dingy dirty jail cell with iron bars, a small cot, and a single, flickering overhead light. A prisoner (donald trump) sits on the cot, head in hands, with a guard standing watch nearby. fantisy, Political Art"
negative_prompt = "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, ."

pipeline = StableDiffusionXLPipeline.from_pretrained(
    model, 
    prompt=prompt,
    negative_prompt=negative_prompt,
    torch_dtype=torch.float16, 
    variant="fp16", 
    add_watermarker=False,
    guidance_scale=guidance_scale,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac
    )
_ = pipeline.to("cuda")

picture = pipeline(prompt)[0][0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
picture.save(f"./output/stable-diffusion-xl-1.0_{timestamp}.png")