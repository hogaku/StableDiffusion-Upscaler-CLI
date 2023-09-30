import base64
import os
import requests
import time
import json
from PIL import Image
import io

class SDImageGenerator:
    def __init__(self, engine_id, api_key, api_host='https://api.stability.ai'):
        self.engine_id = engine_id
        self.api_key = api_key
        self.api_host = api_host

    def generate_image(self, prompt, output_dir, return_image_data=False):
        # Call API
        response = requests.post(
            f"{self.api_host}/v1/generation/{self.engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "text_prompts": [
                    {
                        "text": prompt
                    }
                ],
            },
        )

        if response.status_code != 200:
            raise Exception("Non-200 response: " + str(response.text))

        data = response.json()
        
        resolution_dir = "512_512"
        generated_dir = os.path.join(output_dir, resolution_dir)
        print(generated_dir)
        if not os.path.exists(generated_dir):
            os.makedirs(generated_dir)
        
        pil_image_list = []
        for i, image in enumerate(data["artifacts"]):
            image_data = base64.b64decode(image["base64"])
            with open(f"{generated_dir}/{self.engine_id}_{int(time.time())}_{i}.png", "wb") as f:
                f.write(image_data)
            if return_image_data:
                pil_image = Image.open(io.BytesIO(image_data))
                pil_image_list.append(pil_image)
        
        if return_image_data:
            return pil_image_list

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate images from text.")
    parser.add_argument("engine_id", type=str, default="stable-diffusion-v1-5", help="The engine ID to use for generation.")
    parser.add_argument("prompt", type=str, help="The text prompt for image generation.")
    parser.add_argument("output_folder", type=str, default="./outputs", help="The folder where generated images will be saved.")
    parser.add_argument("--api_key", type=str, help="The API key for the service.")

    args = parser.parse_args()

    generator = SDImageGenerator(args.engine_id, args.api_key)
    generator.generate_image(args.prompt, args.output_folder)
