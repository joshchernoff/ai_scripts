from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
from datetime import datetime
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Directory to save images
IMAGE_DIR = "./output"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Load the models
prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)
prior.enable_model_cpu_offload()
decoder.enable_model_cpu_offload()

def save_image(image, filename):
    image_path = os.path.join(IMAGE_DIR, filename)
    image.save(image_path)
    return image_path

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json

    prompt = data.get("prompt", "Create a portrait of a grim dwarf warrior with a long-braided beard, iron armor, and a mighty axe. Depict them in a Tolkien-esque fantasy world")
    negative_prompt = data.get("negative_prompt", "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck")
    height = int(data.get("height", 1024))
    width = int(data.get("width", 1024))
    guidance_scale_prior = float(data.get("guidance_scale_prior", 4.0))
    guidance_scale_decoder = float(data.get("guidance_scale_decoder", 0.0))
    num_images_per_prompt = int(data.get("num_images_per_prompt", 1))
    num_inference_steps_prior = int(data.get("num_inference_steps_prior", 20))
    num_inference_steps_decoder = int(data.get("num_inference_steps_decoder", 10))
    seed = data.get("seed", None)

    # Set the seed if provided
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None

    # Generate image embeddings
    prior_output = prior(
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale_prior,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps_prior,
        generator=generator
    )

    # Decode the image
    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings.to(torch.float16),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale_decoder,
        output_type="pil",
        num_inference_steps=num_inference_steps_decoder,
        generator=generator
    ).images[0]

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stable-cascade_{timestamp}.png"
    image_path = save_image(decoder_output, filename)

    # Return the URL of the saved image
    image_url = f"http://{request.host}/images/{filename}"
    return jsonify({"image_url": image_url})

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
