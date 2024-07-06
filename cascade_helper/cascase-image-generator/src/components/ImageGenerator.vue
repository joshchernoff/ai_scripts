<template>
  <div class="container mx-auto p-4">
    <div class="flex flex-col lg:flex-row">
      <div class="w-full lg:w-1/2 lg:pr-4 mb-4 lg:mb-0">
        <h1 class="text-2xl font-bold mb-4">Image Generator</h1>
        <form @submit.prevent="generateImage" class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="col-span-1 md:col-span-2">
            <label class="block text-gray-700">Prompt</label>
            <textarea v-model="prompt" class="w-full p-2 border rounded" rows="4"></textarea>
          </div>
          <div class="col-span-1 md:col-span-2">
            <label class="block text-gray-700">Negative Prompt</label>
            <textarea v-model="negativePrompt" class="w-full p-2 border rounded" rows="2"></textarea>
          </div>
          <div>
            <label class="block text-gray-700">Height</label>
            <input type="number" v-model.number="height" class="w-full p-2 border rounded" />
          </div>
          <div>
            <label class="block text-gray-700">Width</label>
            <input type="number" v-model.number="width" class="w-full p-2 border rounded" />
          </div>
          <div>
            <label class="block text-gray-700">Guidance Scale Prior</label>
            <input type="number" v-model.number="guidanceScalePrior" step="0.1" class="w-full p-2 border rounded" />
          </div>
          <div>
            <label class="block text-gray-700">Guidance Scale Decoder</label>
            <input type="number" v-model.number="guidanceScaleDecoder" step="0.1" class="w-full p-2 border rounded" />
          </div>
          <div>
            <label class="block text-gray-700">Number of Images Per Prompt</label>
            <input type="number" v-model.number="numImagesPerPrompt" class="w-full p-2 border rounded" />
          </div>
          <div>
            <label class="block text-gray-700">Number of Inference Steps Prior</label>
            <input type="number" v-model.number="numInferenceStepsPrior" class="w-full p-2 border rounded" />
          </div>
          <div>
            <label class="block text-gray-700">Number of Inference Steps Decoder</label>
            <input type="number" v-model.number="numInferenceStepsDecoder" class="w-full p-2 border rounded" />
          </div>
          <div>
            <label class="block text-gray-700">Seed</label>
            <input type="number" v-model.number="seed" class="w-full p-2 border rounded" />
          </div>
          <div class="col-span-1 md:col-span-2">
            <label class="inline-flex items-center">
              <input type="checkbox" v-model="alwaysGenerateNewSeed" class="form-checkbox">
              <span class="ml-2 text-gray-700">Always generate a new seed</span>
            </label>
          </div>
          <div class="col-span-1 md:col-span-2">
            <button type="submit" class="bg-blue-500 text-white px-4 py-2 rounded w-full">Generate Image</button>
          </div>
        </form>
      </div>
      <div class="w-full lg:w-1/2">
        <h2 v-if="images.length" class="text-xl font-bold mb-4">Generated Images</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div v-for="(image, index) in images" :key="index" class="mb-4">
            <img :src="image.url" alt="Generated Image" class="max-w-full h-auto" />
            <button @click="image.showParams = !image.showParams" class="mt-2 bg-blue-500 text-white px-2 py-1 rounded text-sm">
              {{ image.showParams ? 'Hide Info' : 'Show Info' }}
            </button>
            <div v-if="image.showParams" class="mt-2 text-sm">
              <p><strong>Prompt:</strong> {{ image.params.prompt }}</p>
              <p><strong>Negative Prompt:</strong> {{ image.params.negative_prompt }}</p>
              <p><strong>Height:</strong> {{ image.params.height }}</p>
              <p><strong>Width:</strong> {{ image.params.width }}</p>
              <p><strong>Guidance Scale Prior:</strong> {{ image.params.guidance_scale_prior }}</p>
              <p><strong>Guidance Scale Decoder:</strong> {{ image.params.guidance_scale_decoder }}</p>
              <p><strong>Number of Images Per Prompt:</strong> {{ image.params.num_images_per_prompt }}</p>
              <p><strong>Number of Inference Steps Prior:</strong> {{ image.params.num_inference_steps_prior }}</p>
              <p><strong>Number of Inference Steps Decoder:</strong> {{ image.params.num_inference_steps_decoder }}</p>
              <p><strong>Seed:</strong> {{ image.params.seed }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      prompt: 'Create a portrait of a grim dwarf warrior with a long-braided beard, iron armor, and a mighty axe. Depict them in a Tolkien-esque fantasy world',
      negativePrompt: 'extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck',
      height: 1024,
      width: 1024,
      guidanceScalePrior: 4.0,
      guidanceScaleDecoder: 0.0,
      numImagesPerPrompt: 1,
      numInferenceStepsPrior: 20,
      numInferenceStepsDecoder: 10,
      seed: null,
      alwaysGenerateNewSeed: false,
      images: []
    };
  },
  methods: {
    generateRandomSeed() {
      return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER); // Generates a random seed in a large range
    },
    async generateImage() {
      if (this.alwaysGenerateNewSeed || !this.seed) {
        this.seed = this.generateRandomSeed(); // Generate a random seed if the checkbox is checked or no seed is provided
      }
      const params = {
        prompt: this.prompt,
        negative_prompt: this.negativePrompt,
        height: this.height,
        width: this.width,
        guidance_scale_prior: this.guidanceScalePrior,
        guidance_scale_decoder: this.guidanceScaleDecoder,
        num_images_per_prompt: this.numImagesPerPrompt,
        num_inference_steps_prior: this.numInferenceStepsPrior,
        num_inference_steps_decoder: this.numInferenceStepsDecoder,
        seed: this.seed
      };

      try {
        const response = await axios.post('http://192.168.0.4:5000/generate', params);
        this.images.unshift({
          url: response.data.image_url,
          params: params,
          showParams: false
        }); // Prepend new image with params to the array
      } catch (error) {
        console.error('Error generating image:', error);
      }
    }
  }
};
</script>

<style scoped>
.container {
  width: 100%;
}
</style>
