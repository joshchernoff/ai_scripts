import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "pic_20240705_155256.png"

init_image = load_image(url).convert("RGB")
#prompt = "((masterpiece,best quality)), add detailted background, add detailed forground, add texutre, add highlights, spectual highlights, lensflars"
# prompt = "Create a striking abstract collage using vibrant cutout images of musical instruments. Blend them together in an unusual way with glitch effects."
prompt = "Produce an evocative black-and-white digital landscape featuring rolling foggy hills using only shades of monochrome to highlight light, shadows, and textures for added atmospheric effects"

num_inference_steps = 120 
guidance_scale = 7.0

with torch.no_grad():
    images = pipe(
        prompt,
        image=init_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images

images[0].save("pic_20240705_155256_r.png")
