# GLIDE imports
from typing import Tuple

from IPython.display import display
from PIL import Image
import numpy as np
import torch as th
import torch.nn.functional as F

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

# gradio app imports
import gradio as gr

from torchvision.transforms import ToTensor, ToPILImage
image_to_tensor = ToTensor()
tensor_to_image = ToPILImage()

# This notebook supports both CPU and GPU.
# On CPU, generating one sample may take on the order of 20 minutes.
# On a GPU, it should be under a minute.

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

# Create base model.
options = model_and_diffusion_defaults()
options['inpaint'] = True
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base-inpaint', device))
print('total base parameters', sum(x.numel() for x in model.parameters()))

# Create upsampler model.
options_up = model_and_diffusion_defaults_upsampler()
options_up['inpaint'] = True
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample-inpaint', device))
print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

# Sampling parameters
batch_size = 1
guidance_scale = 5.0

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997

# Create an classifier-free guidance sampling function
def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)

def denoised_fn(x_start):
    # Force the model to have the exact right x_start predictions
    # for the part of the image which is known.
    return (
        x_start * (1 - model_kwargs['inpaint_mask'])
        + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
    )

def show_images(batch: th.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    return Image.fromarray(reshaped.numpy())

def read_image(path: str, size: int = 256) -> Tuple[th.Tensor, th.Tensor]:
    pil_img = Image.open(path).convert('RGB')
    pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

def pil_to_numpy(pil_img: Image) -> Tuple[th.Tensor, th.Tensor]:
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

model_kwargs = dict()
def inpaint(input_img, input_img_with_mask, prompt):
    
    print(prompt)
    
    # Save as png for later mask detection :)
    input_img_256 = input_img.convert('RGB').resize((256, 256), resample=Image.BICUBIC)
    input_img_64 = input_img.convert('RGB').resize((64, 64), resample=Image.BICUBIC)
    
    # Source image we are inpainting
    source_image_256 = pil_to_numpy(input_img_256)
    source_image_64 = pil_to_numpy(input_img_64)
    
    # Since gradio doesn't supply which pixels were drawn, we need to find it ourselves!
    # Assuming that all black pixels are meant for inpainting.
    input_img_with_mask_64 = input_img_with_mask.convert('L').resize((64, 64), resample=Image.BICUBIC)
    gray_scale_source_image = image_to_tensor(input_img_with_mask_64)
    source_mask_64 = (gray_scale_source_image!=0).float()
    source_mask_64_img = tensor_to_image(source_mask_64)
    
    # The mask should always be a boolean 64x64 mask, and then we
    # can upsample it for the second stage.
    source_mask_64 = source_mask_64.unsqueeze(0)
    source_mask_256 = F.interpolate(source_mask_64, (256, 256), mode='nearest')
    
    
    ##############################
    # Sample from the base model #
    ##############################

    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    global model_kwargs
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),

        # Masked inpainting image
        inpaint_image=(source_image_64 * source_mask_64).repeat(full_batch_size, 1, 1, 1).to(device),
        inpaint_mask=source_mask_64.repeat(full_batch_size, 1, 1, 1).to(device),
    )
    
    # Sample from the base model.
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    model.del_cache()
    
    ##############################
    # Upsample the 64x64 samples #
    ##############################

    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up['text_ctx']
    )

    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,

        # Text tokens
        tokens=th.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),

        # Masked inpainting image.
        inpaint_image=(source_image_256 * source_mask_256).repeat(batch_size, 1, 1, 1).to(device),
        inpaint_mask=source_mask_256.repeat(batch_size, 1, 1, 1).to(device),
    )

    # Sample from the base model.
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.p_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    model_up.del_cache()
    
    return source_mask_64_img, show_images(up_samples)

gradio_inputs = [gr.inputs.Image(type='pil', 
                                 label="Input Image"),
                 gr.inputs.Image(type='pil', 
                                 label="Input Image With Mask"),
                 gr.inputs.Textbox(label='Conditional Text to Inpaint')]

# gradio_outputs = [gr.outputs.Image(label='Auto-Detected Mask (From drawn black pixels)')]

gradio_outputs = [gr.outputs.Image(label='Auto-Detected Mask (From drawn black pixels)'),
                 gr.outputs.Image(label='Inpainted Image')]
examples = [['grass.png', 'grass_with_mask.png', 'a corgi in a field']]

title = "GLIDE Inpaint"
description = "Using GLIDE to inpaint black regions of an input image! Instructions: 1) For the 'Input Image', upload an image. 2) For the 'Input Image with Mask', draw a black-colored mask (either manually with something like Paint, or by using gradio's built-in image editor & add a black-colored shape) IT MUST BE BLACK COLOR, but doesn't have to be rectangular! This is because it auto-detects the mask based on 0 (black) pixel values! 3) For the Conditional Text, type something you'd like to see the black region get filled in with :)"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2112.10741' target='_blank'>GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models</a> | <a href='https://github.com/openai/glide-text2im' target='_blank'>Github Repo</a></p>"
iface = gr.Interface(fn=inpaint, inputs=gradio_inputs,
                     outputs=gradio_outputs,
                     examples=examples, title=title,
                     description=description, article=article,
                     enable_queue=True)

iface.launch(share=True, enable_queue=True)
