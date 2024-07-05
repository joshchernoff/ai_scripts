from diffusers import StableDiffusionPipeline
import torch

def main():
    # Load the pretrained Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    # Generate an image from text
    prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
    image = pipe(prompt).images[0]

    # Save the image
    image.save("stable-diffusion-v1-4.png")
    print("Image saved as output.png")

if __name__ == "__main__":
    main()

