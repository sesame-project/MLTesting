# # run this on terminal huggingface-cli login
# from io import BytesIO
# from matplotlib import gridspec
# from matplotlib import pyplot as plt
# # from torch import autocast
# from torch.cuda.amp import autocast
# import requests
# import PIL
# import torch
# from diffusers import StableDiffusionInpaintPipeline
#
# # from inpainting import StableDiffusionInpaintPipeline
# def download_image(url):
#     response = requests.get(url)
#     return PIL.Image.open(BytesIO(response.content)).convert("RGB")
# def vis_segmentation(inimage, outimg):
#   """Visualizes input image, segmentation map and overlay view."""
#   plt.figure(figsize=(15, 5))
#   grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
#
#   plt.subplot(grid_spec[0])
#   plt.imshow(inimage)
#   plt.axis('off')
#   plt.title('input image')
#
#
#
#   plt.subplot(grid_spec[0])
#   plt.imshow(outimg)
#
#   plt.axis('off')
#   plt.title('augmented image')
#
#
#
#
#
# if __name__ == "__main__":
#     access_token='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'
#
#     img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
#     mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
#     init_image = download_image(img_url).resize((512, 512))
#     mask_image = download_image(mask_url).resize((512, 512))
#     # device = "cuda"
#     # pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     #     "CompVis/stable-diffusion-v1-4", revision="fp16",
#     #     torch_dtype=torch.float16,
#     #     use_auth_token=access_token)
#     pipe = StableDiffusionInpaintPipeline.from_pretrained(
#         "CompVis/stable-diffusion-v1-4",
#         torch_dtype=torch.float16,
#         use_auth_token=access_token)
#     # pipe = pipe.to("cuda")
#     # pipe = pipe.to("cpu")
#
#
#     prompt = "a kungfu panda sitting on a bench"
#     # with autocast("cuda"):
#     images = pipe(prompt=prompt, init_image=init_image, mask_image=mask_image, strength=0.75)["sample"]
#     images[0].save("cat_on_bench.png")
#     # print(images[0])
#     vis_segmentation(init_image, images[0])




# url = 'https://raw.githubusercontent.com/huggingface/diffusers/main/examples/inference/inpainting.py'
# filename = wget.download(url)
# print(filename)
from io import BytesIO
# from torch import autocast
# from torch.cuda.amp import autocast
# from stable_diffusion_tf.stable_diffusion import Text2Image
import requests
import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline
# from inpainting import StableDiffusionInpaintPipeline
import cv2
from PIL import Image
HF_DATASETS_OFFLINE = 1
TRANSFORMERS_OFFLINE = 1
def Inpainting(Init_img, mask, augmented_caption):

        access_token = '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'
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        # if device.type != 'cuda':
        #     raise ValueError("need to run on GPU")
        device = "cuda"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=access_token).to(device)

        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #         "CompVis/stable-diffusion-v1-4",
        #         torch_dtype=torch.float16,
        #         use_auth_token=access_token)
        init_image = Image.open(Init_img).resize((512, 512))
        mask_image = Image.open(mask).resize((512, 512))
        #
        # with autocast("cuda"):
        with torch.no_grad():
        # with torch.cuda.amp.autocast(True):
            images = pipe(prompt=augmented_caption, init_image=init_image, mask_image=mask_image, strength=0.75).images
        # images[0].save(path+"result.png")
        return images

