from typing import Any

import base64
import gzip
import random
import subprocess
import urllib.request
import cv2
import torch
from omegaconf import OmegaConf
import numpy as np
import os
from basicsr.utils import imwrite
import traceback

from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange, repeat
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from gfpgan import GFPGANer

import uuid
import json
import sys
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote

from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

# GLOBAL CONSTS
# Notes from https://github.com/pesser/stable-diffusion/blob/main/README.md
# Quality, sampling speed and diversity are best controlled via the scale, DDIM_STEPS and DDIM_ETA arguments.
# As a rule of thumb, higher values of scale produce better samples at the cost of a reduced output diversity.
# Furthermore, increasing DDIM_STEPS generally also gives higher quality samples,
# but returns are diminishing for values > 250.
# Fast sampling (i.e. low values of ddim_steps) while retaining good quality can be achieved by using --ddim_eta 0.0.
# Faster sampling (i.e. even lower values of DDIM_STEPS) while retaining good quality can be achieved
# by using DDIM_ETA 0.0 (only PLMS used in this app).

LATENT_CHANNELS = 4  # was opc.C - can't change that because that's inherent in the trained model.
HEIGHT = 512
WIDTH = 512
DOWNSAMPLING_FACTOR = 8  # was opt.F
SAMPLE_THIS_OFTEN = 1  # was opt.n_iter
SD_MODEL_PATH = 'models/ldm/stable-diffusion-v1/model.ckpt'
INPAINTING_MODEL_PATH = 'models/ldm/stable-diffusion-v1/last.ckpt'
SCALE = 7.5  # was opt.scale#
DDIM_STEPS = 40  # was opt.ddim_steps (number of ddim sampling steps)
DDIM_ETA = 0.0  # was opt.ddim_eta  (ddim eta (eta=0.0 corresponds to deterministic sampling)
N_SAMPLES = 1  # was opt.n_samples (how many samples to produce for each given prompt. A.k.a. batch size)
PRECISION = "autocast"  # can be "autocast" or "full"
STRENGTH = 0.75  # was opt.strength - used when processing an image - 0 means no change through 0.999 means full change
OUTPUT_PATH = '/library'
PORT = 8080
WATERMARK_FLAG = False  # set to True to enable watermarking
SAFETY_FLAG = False  # set to True to enable safety checking - a 'NSFW' image will be returned if the safety check fails

# IMAGE QUALITY SETTINGS
IMAGE_QUALITY = "HIGH"
# "MAX" - maximum quality with PNG format - 420-500 Kb per image, upscale 4MB-5MB
# "HIGH" - high quality with JPG format - 120-200 Kb per image, upscale 1-2 MB
# "LOW" - lower/medium quality with JPG format - 15-30 Kb per image, upscale 400-800 Kb

# GLOBAL VARS
global_device = None
global_model = None
global_wm_encoder = None


def format_exception(e):
    exception_list = traceback.format_stack()
    exception_list = exception_list[:-2]
    exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
    exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))

    exception_str = "Traceback (most recent call last):\n"
    exception_str += "".join(exception_list)
    # Removing the last \n
    exception_str = exception_str[:-1]

    return exception_str

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_and_format_image(path, output_width, output_height, zoom_factor=1.0):
    # load an image from a URL
    if path.startswith("http"):
        # get image name from URL
        image_name = path.split("/")[-1]
        print(f"Loading image from web {path} to save as {image_name}")
        urllib.request.urlretrieve(path, image_name)
        image = Image.open(image_name).convert("RGB")
    # load an image from a file
    elif path.startswith('library'):
        print(f"Loading image from volume path {path}")
        image = Image.open('/' + path).convert("RGB")
    else:
        print(f"Loading image from volume path {path}")
        image = Image.open(path).convert("RGB")

    print(f"loaded input image from {path}")

    # resize image to fit model, maintaining aspect ratio
    w, h = image.size
    print(f"image size ({w}, {h})")

    old_size = image.size  # old_size[0] is in (width, height) format

    if output_width > output_height:
        ratio = float(output_width) / max(old_size)
    else:
        ratio = float(output_height) / max(old_size)

    new_size = tuple([int(x * ratio * zoom_factor) for x in old_size])

    resized_image = image.resize(new_size, Image.ANTIALIAS)

    resized_w, resized_h = resized_image.size
    print(f"Image resized to size ({resized_w}, {resized_h}) ")

    # create a new image and paste the resized on it. The new image size is the same size as the
    # requested output. The resized image is centered in the new image maintaining aspect ratio.
    # If it's not an exact fit, black bands will appear at the top/bottom or sides of the new image.
    # That's not an issue but the app will only fill in the black parts if the image can be changed
    # by more than 80% (see STRENGTH parameter)
    new_image = Image.new("RGB", (output_width, output_height))
    new_image.paste(resized_image, ((output_width - new_size[0]) // 2, (output_height - new_size[1]) // 2))

    image = np.array(new_image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    processed_image = 2. * image - 1.

    # ImageDraw.Draw(new_image).text((10, 10), "Original Image", fill=(255, 255, 255))
    # line above commented out because burning the words "Original Image" into the image was disrupting the 'advanced'
    # workflow where I take an image from the libray to manipulate it further. You should know the original image!

    return new_image, processed_image


def put_watermark(img, wm_encoder=None):
    if WATERMARK_FLAG:
        if wm_encoder is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = wm_encoder.encode(img, 'dwtDct')
            img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/nsfw.png").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def danger_will_robinson(x_image):
    return x_image, []


def setup():
    print("Setting up model ready for inference")

    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, 'models/ldm/stable-diffusion-v1/model.ckpt')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    return device, model, wm_encoder


def process(text_prompt, device, model, wm_encoder, queue_id, num_images, options, action="prompt"):
    print('Running Prompt Processing')

    chosen_seed = options['seed']
    if chosen_seed == 0:
        chosen_seed = random.randint(1, 2147483647)
    else:
        # we push the seed back by 1 as it will be incremented as we start the looping
        chosen_seed -= 1

    # NEW! Ability to process negative prompts
    if 'negative_prompt' in options:
        print("DEBUG: negative prompt found")
        negative_prompt = options['negative_prompt']
    else:
        print("DEBUG: NO negative prompt found")
        negative_prompt = ""

    sampler = PLMSSampler(model)  # Uses PLMS model
    start = time.time()
    library_dir_name = os.path.join(OUTPUT_PATH, queue_id)
    os.makedirs(library_dir_name, exist_ok=True)
    image_counter = 0

    try:
        assert text_prompt is not None
        data = [N_SAMPLES * [text_prompt]]
        start_code = None
        fixed_code = False  # but these may be in  advanced input parameters in future
        if fixed_code:
            start_code = torch.randn(
                [N_SAMPLES, LATENT_CHANNELS, options['height'] // options['downsampling_factor'],
                 options['width'] // options['downsampling_factor']],
                device=device)

        precision_scope = autocast if PRECISION == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for n in trange(int(num_images / N_SAMPLES), desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            unconditional_conditioning = None
                            if SCALE != 1.0:
                                unconditional_conditioning = model.get_learned_conditioning(
                                    N_SAMPLES * [negative_prompt])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            conditioning = model.get_learned_conditioning(prompts)

                            shape = [LATENT_CHANNELS, options['height'] // options['downsampling_factor'],
                                     options['width'] // options['downsampling_factor']]

                            max_ddim_steps = options['max_ddim_steps']
                            min_ddim_steps = options['min_ddim_steps']
                            print(f"Running DDIM steps {min_ddim_steps} to {max_ddim_steps}")
                            for each_ddim_step in range(min_ddim_steps, max_ddim_steps + 1):
                                print(f"Running DDIM step {each_ddim_step}")

                                if max_ddim_steps == min_ddim_steps:
                                    # We're going to increment the seed counter for each image in order to
                                    # document the seed used at the image level - the seed value will be recorded
                                    # in the image file name.
                                    # if the steps differ then we keep the same seed for consistency for the 'one' image
                                    chosen_seed += 1

                                print(f"Running DDIM step {each_ddim_step} with seed {chosen_seed}")
                                seed_everything(chosen_seed)

                                image_counter, first_image_path = run_sampling(image_counter,
                                                                               conditioning,
                                                                               each_ddim_step,
                                                                               library_dir_name,
                                                                               model,
                                                                               options,
                                                                               sampler,
                                                                               shape,
                                                                               start_code,
                                                                               unconditional_conditioning,
                                                                               wm_encoder,
                                                                               chosen_seed)

                            end = time.time()
                            time_taken = end - start
                            image_counter += 1

                    if action == "prompt":
                        save_metadata_file(num_images, library_dir_name, options, queue_id, text_prompt, time_taken, '',
                                           '')

        return {'success': True, 'queue_id': queue_id, 'first_image_path': first_image_path}

    except Exception as e:
        print(e)
        end = time.time()
        time_taken = end - start
        save_metadata_file(num_images, library_dir_name, options, queue_id, text_prompt, time_taken, str(e), '')
        return {'success': False, 'error: ': 'error: ' + str(e), 'queue_id': queue_id}


def run_sampling(image_counter, conditioning, ddim_steps, library_dir_name, model, options, sampler, shape, start_code,
                 unconditional_conditioning, wm_encoder, seed_value):
    first_image_path = ''
    try:
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                         conditioning=conditioning,
                                         batch_size=N_SAMPLES,
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=options['scale'],
                                         unconditional_conditioning=unconditional_conditioning,
                                         eta=options['ddim_eta'],
                                         x_T=start_code)
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

        if SAFETY_FLAG:
            x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
        else:
            x_checked_image, has_nsfw_concept = danger_will_robinson(x_samples_ddim)

        x_samples = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

        image_counter, first_image_path = save_image_samples(ddim_steps, image_counter, library_dir_name, wm_encoder,
                                                             x_samples,
                                                             seed_value, options['scale'])

    except Exception as e:
        print('Error in run_sampling: ' + str(e))

    return image_counter, first_image_path


def upscale_image(image_list, queue_id, upscale_factor=2):
    # replacements for default arguments
    bg_upsampler = "realesrgan"
    bg_tile = 200
    version = '1.4'  # I've included "1.3" model in this repo as well
    upscale = upscale_factor
    file_suffix = "upscaled"  # The new image will sit alongside the original in the library with a '_upscaled' suffix
    only_centre_face = False
    aligned_faces = False
    weight = 0.5

    response = {'success': False, 'queue_id': queue_id}

    os.chdir('/app/GFPGAN')
    try:
        # ------------------------ set up background upsampler ------------------------
        if bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                import warnings
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                              'If you really want to use it, please modify the corresponding codes.')
                bg_upsampler = None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='/usr/local/lib/python3.10/dist-packages/realesrgan/weights/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)  # need to set False in CPU mode
        else:
            bg_upsampler = None

        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'

        # determine model paths
        model_path = os.path.join('/app/GFPGAN/experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            # download pre-trained models from url
            model_path = url

        restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)

        # ------------------------ restore ------------------------
        for img_path in image_list:

            if img_path.startswith('library/'):
                img_path = "/" + img_path

            # read image
            img_name = os.path.basename(img_path)
            output_path = os.path.dirname(img_path)
            print(f'Processing {img_name} ...')
            basename, ext = os.path.splitext(img_name)
            input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # restore faces and background if necessary
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                input_img,
                has_aligned=aligned_faces,
                only_center_face=only_centre_face,
                paste_back=True,
                weight=weight)

            # save restored img
            if restored_img is not None:
                if IMAGE_QUALITY == "MAX":
                    save_restore_path = os.path.join(output_path, f'{basename}_{file_suffix}.png')
                else:
                    save_restore_path = os.path.join(output_path, f'{basename}_{file_suffix}.jpg')

                imwrite(restored_img, save_restore_path)
                response['success'] = True
            else:
                print('No image to restore!')


    except Exception as e:
        print("Error", e)

    os.chdir('/app')
    return response


def save_image_samples(ddim_steps, image_counter, library_dir_name, wm_encoder, x_samples, seed_value, scale,
                       video_frame_number=0):
    # Saves the image samples in format: <image_counter>_D<ddim_steps>_S<scale>_R<seed_value>-<random 8 characters>.png/.jpg
    first_image_path = ''
    for x_sample in x_samples:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img = put_watermark(img, wm_encoder)

        if video_frame_number == 0:
            print(f"Saving image_number: {image_counter + 1}")
            image_name = f"{image_counter + 1:03d}-D{ddim_steps:03d}-S{scale:.1f}-R{seed_value:0>4}-{str(uuid.uuid4())[:8]}"
        else:
            print(f"Saving video frame number: {video_frame_number}")
            image_name = f"{video_frame_number:04d}-D{ddim_steps:03d}-S{scale:.1f}-R{seed_value:0>4}-{str(uuid.uuid4())[:8]}"

        if IMAGE_QUALITY == "MAX":
            if first_image_path == '':
                first_image_path = os.path.join(library_dir_name, image_name + ".png")
            img.save(os.path.join(library_dir_name, image_name + ".png"), optimize=True, progressive=True)

        elif IMAGE_QUALITY == "HIGH":
            if first_image_path == '':
                first_image_path = os.path.join(library_dir_name, image_name + ".jpg")
            img.save(os.path.join(library_dir_name, image_name + ".jpg"), quality='web_maximum')

        elif IMAGE_QUALITY == "LOW":
            if first_image_path == '':
                first_image_path = os.path.join(library_dir_name, image_name + ".jpg")
            img.save(os.path.join(library_dir_name, image_name + ".jpg"), quality='web_low')

    if video_frame_number == 0:
        image_counter += 1
        return image_counter, first_image_path
    else:
        return video_frame_number, first_image_path


def process_image(original_image_path, text_prompt, device, model, wm_encoder, queue_id, num_images, options,
                  video_frame_number=0, zoom_factor=1.0):
    print('Running Image Processing')
    sampler = DDIMSampler(model)  # Uses DDIM model

    chosen_seed = options['seed']
    if chosen_seed == 0:
        chosen_seed = random.randint(1, 2147483647)
    else:
        # we push the seed back by 1 as it will be incremented as we start the looping
        chosen_seed -= 1

    # NEW! Negative prompting
    if 'negative_prompt' in options:
        negative_prompt = options['negative_prompt']
    else:
        negative_prompt = ''

    start = time.time()
    library_dir_name = os.path.join(OUTPUT_PATH, queue_id)
    os.makedirs(library_dir_name, exist_ok=True)
    image_counter = 0

    try:
        assert text_prompt is not None
        data = [N_SAMPLES * [text_prompt]]

        # load the image
        resized_image, init_image = load_and_format_image(original_image_path,
                                                          options['width'], options['height'],
                                                          zoom_factor=zoom_factor)

        # save the resized original image
        if IMAGE_QUALITY == "MAX":
            resized_image.save(os.path.join(library_dir_name, '00-original.png'))
        elif IMAGE_QUALITY == "HIGH":
            resized_image.save(os.path.join(library_dir_name, '00-original.jpg'), quality='web_high')
        elif IMAGE_QUALITY == "LOW":
            resized_image.save(os.path.join(library_dir_name, '00-original.jpg'), quality='web_low')

        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=N_SAMPLES)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        max_ddim_steps = options['max_ddim_steps']
        min_ddim_steps = options['min_ddim_steps']
        print(f"Running DDIM steps {min_ddim_steps} to {max_ddim_steps}")

        for each_ddim_step in range(min_ddim_steps, max_ddim_steps + 1):
            try:
                print(f"Running DDIM step {each_ddim_step}")
                seed_everything(options['seed'])

                sampler.make_schedule(ddim_num_steps=each_ddim_step, ddim_eta=DDIM_ETA, verbose=False)

                assert 0. <= options['strength'] <= 1., 'can only work with strength in [0.0, 1.0]'

                t_enc = int(options['strength'] * each_ddim_step)
                print(f"target t_enc is {t_enc} steps")

                precision_scope = autocast if PRECISION == "autocast" else nullcontext
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            for n in trange(int(num_images / N_SAMPLES), desc="Sampling"):
                                for prompts in tqdm(data, desc="data"):
                                    if max_ddim_steps == min_ddim_steps:
                                        # increment the seed counter so we can keep control of the seed for each image
                                        # (if the steps differ then we keep the same seed for consistency
                                        # for the 'one' image)
                                        chosen_seed += 1

                                    print(f"Running DDIM step {each_ddim_step} with seed {chosen_seed}")
                                    seed_everything(chosen_seed)

                                    uc = None
                                    if SCALE != 1.0:
                                        uc = model.get_learned_conditioning(N_SAMPLES * [negative_prompt])
                                    if isinstance(prompts, tuple):
                                        prompts = list(prompts)

                                    c = model.get_learned_conditioning(prompts)

                                    # encode (scaled latent)
                                    z_enc = sampler.stochastic_encode(init_latent,
                                                                      torch.tensor([t_enc] * N_SAMPLES).to(device))
                                    # decode it
                                    samples = sampler.decode(z_enc, c, t_enc,
                                                             unconditional_guidance_scale=options['scale'],
                                                             unconditional_conditioning=uc, )

                                    x_samples = model.decode_first_stage(samples)
                                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                    # save the newly created images
                                    image_counter, first_image_path = save_image_samples(each_ddim_step,
                                                                                         image_counter,
                                                                                         library_dir_name,
                                                                                         wm_encoder,
                                                                                         x_samples,
                                                                                         chosen_seed,
                                                                                         options['scale'],
                                                                                         video_frame_number)

                                    end = time.time()
                                    time_taken = end - start

            except Exception as e:
                print('Error in process_image: ' + str(e))

        if video_frame_number == 0:
            # We are saving a conventional image metadate index file
            save_metadata_file(num_images=image_counter,
                               library_dir_name=library_dir_name,
                               options=options,
                               queue_id=queue_id,
                               text_prompt=text_prompt,
                               time_taken=time_taken,
                               error='',
                               original_image_path=original_image_path,
                               format='image')

        return {'success': True, 'queue_id': queue_id, 'first_image_path': first_image_path}

    except Exception as e:
        print(e)
        end = time.time()
        time_taken = end - start
        if video_frame_number > 0:
            save_metadata_file(num_images=0,
                               library_dir_name=library_dir_name,
                               options=options,
                               queue_id=queue_id,
                               text_prompt=text_prompt,
                               time_taken=time_taken,
                               error=str(e),
                               original_image_path=original_image_path,
                               zoom_factor=zoom_factor,
                               format='video')

        else:
            save_metadata_file(num_images=0,
                               library_dir_name=library_dir_name,
                               options=options,
                               queue_id=queue_id,
                               text_prompt=text_prompt,
                               time_taken=time_taken,
                               error=str(e),
                               original_image_path=original_image_path,
                               format='image')

        return {'success': False, 'error: ': 'error: ' + str(e), 'queue_id': queue_id}


def check_api_request_properties(data, command):
    options = {
        'seed': 0,
        'height': HEIGHT,
        'width': WIDTH,
        'max_ddim_steps': DDIM_STEPS,
        'min_ddim_steps': DDIM_STEPS,
        'ddim_eta': DDIM_ETA,
        'scale': SCALE,
        'downsampling_factor': DOWNSAMPLING_FACTOR,
        'strength': STRENGTH,
        'negative_prompt': ''
    }

    # Override the default options with any in the request:
    try:
        if 'seed' in data and len(str(data['seed']).strip()) > 0 and int(data['seed']) > 0:
            options['seed'] = int(data['seed'])
        # else use the seed generated when options was initialised

    except ValueError:
        # keep the seed at 0 to be generated within the process functions
        options['seed'] = 0

    if 'height' in data:
        options['height'] = int(data['height'])

    if 'width' in data:
        options['width'] = int(data['width'])

    if 'ddim_eta' in data:
        options['ddim_eta'] = float(data['ddim_eta'])

    if 'scale' in data:
        options['scale'] = float(data['scale'])

    if 'downsampling_factor' in data:
        options['downsampling_factor'] = int(data['downsampling_factor'])

    if 'zoom_factor' in data:
        options['zoom_factor'] = float(data['zoom_factor'])

    if 'ddim_steps' in data:
        options['max_ddim_steps'] = int(data['ddim_steps'])

    if 'max_ddim_steps' in data:
        options['max_ddim_steps'] = int(data['max_ddim_steps'])

    if 'min_ddim_steps' in data:
        options['min_ddim_steps'] = int(data['min_ddim_steps'])

    if options['min_ddim_steps'] > options['max_ddim_steps'] or command == 'video':
        options['min_ddim_steps'] = options['max_ddim_steps']

    if 'negative_prompt' in data:
        options['negative_prompt'] = data['negative_prompt']

    if 'strength' in data:
        options['strength'] = float(data['strength'])
        if options['strength'] >= 1.0:
            options['strength'] = 0.999
        elif options['strength'] < 0.0:
            options['strength'] = 0.001

    original_image_path = ''
    if 'original_image_path' in data:
        original_image_path = data['original_image_path'].strip()

        if original_image_path.startswith('library/'):
            original_image_path = original_image_path.replace('library/', '/library/')

        if not (original_image_path.startswith('http') or original_image_path.startswith('/library/')):
            if original_image_path != '':
                print('Warning: "{}" is not a valid IMAGE URL - processing continues as if no file present'.format(
                    original_image_path))
            original_image_path = ''

    options['original_image_path'] = original_image_path

    original_mask_path = ''
    if 'original_mask_path' in data:
        original_mask_path = data['original_mask_path'].strip()

        if original_mask_path.startswith('library/'):
            original_mask_path = original_mask_path.replace('library/', '/library/')

        if not (original_mask_path.startswith('http') or original_mask_path.startswith('/library/')):
            if original_mask_path != '':
                print('Warning: "{}" is not a valid MASK URL - processing continues as if no file present'.format(
                    original_mask_path))
            original_mask_path = ''

    options['original_mask_path'] = original_mask_path

    return options


def save_metadata_file(num_images, library_dir_name, options, queue_id, text_prompt, time_taken, error,
                       original_image_path, zoom_factor=0.0, format="image"):
    with open(library_dir_name + '/index.json', 'w', encoding="utf8") as outfile:
        metadata = {
            "text_prompt": text_prompt,
            "negative_prompt": options['negative_prompt'],
            "num_images": num_images,
            "queue_id": queue_id,
            "time_taken": round(time_taken, 2),
            "seed": options['seed'],
            "height": options['height'],
            "width": options['width'],
            "min_ddim_steps": options['min_ddim_steps'],
            "max_ddim_steps": options['max_ddim_steps'],
            "ddim_eta": options['ddim_eta'],
            "scale": options['scale'],
            "downsampling_factor": options['downsampling_factor'],
            "error": error,
            "original_image_path": original_image_path,
            "strength": options['strength'] if "strength" in options else -1  # -1 means not applicable
        }
        json.dump(metadata, outfile, indent=4, ensure_ascii=False)


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Backend server running!')

    def do_POST(self):
        api_command = unquote(self.path)
        print("\nPOST >> API command =", api_command)

        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body.decode('utf-8'))

        if type(data) is not dict:
            data = json.loads(data)

        print("\nIncoming POST request data =", data)

        if api_command == '/prompt':
            self.process_prompt(data)
        elif api_command == '/upscale':
            self.process_upscale(data)
        elif api_command == '/video':
            self.process_video(data)
        elif api_command == "/inpaint":
            self.process_inpaint(data)

    def process_upscale(self, data):
        image_list = data['image_list']
        upscale_factor = data['upscale_factor']
        queue_id = data['queue_id']
        response = upscale_image(image_list, queue_id, upscale_factor)
        if response['success']:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
        else:
            self.send_response(500)
            self.end_headers()

        response_body = json.dumps(response)
        self.wfile.write(response_body.encode())

    def inpaint_make_batch(self, image, mask, device):
        image = np.array(Image.open(image).convert("RGB"))
        image = image.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        mask = np.array(Image.open(mask).convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = (1 - mask) * image

        batch = {"image": image, "mask": mask, "masked_image": masked_image}
        for k in batch:
            batch[k] = batch[k].to(device=device)
            batch[k] = batch[k] * 2.0 - 1.0
        return batch

    def process_inpaint(self, data):
        print('Running Inpaint Processing')
        image_path = ''
        mask_path = ''

        try:
            queue_id = data['queue_id']
            num_images = data['num_images']
        except KeyError as e:
            print(e)
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Bad request: missing prompt, image_list, queue_id or num_images"}')
            return

        options = check_api_request_properties(data, "prompt")

        if options['original_image_path'] == '':
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Bad request: missing original_image_path", "queue_id": []}')
            return
        else:
            image_path = options['original_image_path']
            print('image_path', image_path)

        if options['original_mask_path'] == '':
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Bad request: missing original_mask_path", "queue_id": []}')
            return
        else:
            mask_path = options['original_mask_path']
            print('mask_path', mask_path)

        chosen_seed = data['seed']
        if chosen_seed == 0:
            chosen_seed = random.randint(1, 2147483647)
        else:
            # we push the seed back by 1 as it will be incremented as we start the looping
            chosen_seed -= 1

        start = time.time()
        library_dir_name = os.path.join(OUTPUT_PATH, queue_id)
        os.makedirs(library_dir_name, exist_ok=True)
        image_counter = 0

        config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
        model = instantiate_from_config(config.model)
        # model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"], strict=False)
        model.load_state_dict(torch.load(INPAINTING_MODEL_PATH)["state_dict"], strict=False)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        sampler = DDIMSampler(model)

        ddim_steps = options['max_ddim_steps']

        # create the path if it does not already exist
        library_dir_name = os.path.join(OUTPUT_PATH, queue_id)
        os.makedirs(library_dir_name, exist_ok=True)

        # Resize the image to the correct size for internal processing
        print('Resizing image to {}px width x {}px height'.format(options['width'], options['height']))
        resized_image, init_image = load_and_format_image(image_path, options['width'], options['height'])
        resized_image.save(image_path)

        # Resize the mask to the correct size for internal processing
        print('Resizing mask to {}px width x {}px height'.format(options['width'], options['height']))
        resized_mask, init_mask = load_and_format_image(mask_path, options['width'], options['height'])
        resized_mask.save(mask_path)

        # Now make an inpaint natch from the image and mask
        batch = self.inpaint_make_batch(image_path, mask_path, device=device)

        try:
            with torch.no_grad():
                with model.ema_scope():
                    for image_number in range(num_images):

                        # encode masked image and concat downsampled mask
                        c = model.cond_stage_model.encode(batch["masked_image"])
                        cc = torch.nn.functional.interpolate(batch["mask"], size=c.shape[-2:])
                        c = torch.cat((c, cc), dim=1)

                        shape = (c.shape[1] - 1,) + c.shape[2:]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=c.shape[0],
                                                         shape=shape,
                                                         verbose=False)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)

                        image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
                        mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
                        predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        inpainted = (1 - mask) * image + mask * predicted_image
                        inpainted_transposed = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255

                        # Generate output filename
                        out_filename = "{}/inpainted_{}".format(library_dir_name, image_number)
                        if IMAGE_QUALITY == "MAX":
                            out_filename += ".png"
                        else:
                            out_filename += ".jpg"

                        print("Saving image to {}".format(out_filename))
                        Image.fromarray(inpainted_transposed.astype(np.uint8)).save(out_filename)

            end = time.time()
            time_taken = end - start
            save_metadata_file(num_images, library_dir_name, options, queue_id, "INPAINTED", original_image_path=image_path, time_taken=time_taken, error='')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response_body = json.dumps({'success': True, 'queue_id': queue_id})
            self.wfile.write(response_body.encode())

        except ValueError as ve:
            print('process_inpaint ValueError:', ve, format_exception(ve))
            self.send_response(500)
            self.end_headers()
            response_body = json.dumps({'success': False, 'queue_id': queue_id, 'error': repr(ve)})
            self.wfile.write(response_body.encode())

        except Exception as e:
            print('process_inpaint Exception:', e,format_exception(e))
            self.send_response(500)
            self.end_headers()
            response_body = json.dumps({'success': False, 'queue_id': queue_id, 'error': repr(e)})
            self.wfile.write(response_body.encode())

    def process_prompt(self, data):
        # Get the mandatory prompt data from the request
        try:
            prompt = data['prompt'].strip()
            queue_id = data['queue_id']
            num_images = data['num_images']
        except KeyError as e:
            print(e)
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Bad request: missing prompt, queue_id or num_images"}')
            return

        options = check_api_request_properties(data, "prompt")

        # process!
        if options['original_image_path'] != '':
            result = process_image(options['original_image_path'], prompt, global_device, global_model,
                                   global_wm_encoder, queue_id,
                                   num_images, options)
        else:
            result = process(prompt, global_device, global_model, global_wm_encoder, queue_id,
                             num_images, options)

        # Send the response back to the calling request
        if result == 'X':
            self.send_response(500)
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response_body = json.dumps(result)
            self.wfile.write(response_body.encode())

    def process_video(self, data):
        # Get the mandatory prompt data from the request
        try:
            prompt = data['prompt'].strip()
            queue_id = data['queue_id']
            num_video_frames = data['num_video_frames']
            num_frames_per_second = data['frames_per_second']
            zoom_factor = data['zoom_factor']

        except KeyError as e:
            print(e)
            self.send_response(400)
            self.end_headers()
            self.wfile.write(
                b'{"error": "Bad request: missing prompt, queue_id, num_video_frames, or num_frames_per_second"}')
            return

        options = check_api_request_properties(data, "video")

        video_frame_paths_list = []
        video_frame_count = 0
        start = time.time()

        # First run with or without an original image to get the first frame
        if options['original_image_path'] != '':
            result = process_image(original_image_path=options['original_image_path'],
                                   text_prompt=prompt,
                                   device=global_device,
                                   model=global_model,
                                   wm_encoder=global_wm_encoder,
                                   queue_id=queue_id,
                                   num_images=1,
                                   options=options,
                                   zoom_factor=1.0)
        else:
            result = process(text_prompt=prompt,
                             device=global_device,
                             model=global_model,
                             wm_encoder=global_wm_encoder,
                             queue_id=queue_id,
                             num_images=1,
                             options=options,
                             action="video")

        video_frame_paths_list.append(result['first_image_path'])

        # loop through the video frames, creating a new image based on the previous image
        for video_frame_count in range(1, num_video_frames):
            print("### Creating video frame {}/{}".format(video_frame_count + 1, num_video_frames))
            result = process_image(original_image_path=result['first_image_path'],
                                   text_prompt=prompt,
                                   device=global_device,
                                   model=global_model,
                                   wm_encoder=global_wm_encoder,
                                   queue_id=queue_id,
                                   num_images=1,
                                   options=options,
                                   video_frame_number=video_frame_count,
                                   zoom_factor=zoom_factor)

            print("### Video frame result", result)
            video_frame_paths_list.append(result['first_image_path'])

        # Now use ffmpeg to create the video
        video_path = create_video_from_frames(video_frame_paths_list, queue_id, num_frames_per_second)
        del result['first_image_path']
        result['video_path'] = video_path

        end = time.time()
        time_taken = end - start

        # Saving the metadata file here is important, as it signals to the client that
        # processing is complete and the video is ready
        save_metadata_file(num_images=video_frame_count,
                           library_dir_name=os.path.join(OUTPUT_PATH, queue_id),
                           options=options,
                           queue_id=queue_id,
                           text_prompt=prompt,
                           time_taken=time_taken,
                           error='',
                           original_image_path=options['original_image_path'],
                           zoom_factor=zoom_factor,
                           format='video')

        # Send the response back to the calling request
        if result == 'X':
            self.send_response(500)
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response_body = json.dumps(result)
            self.wfile.write(response_body.encode())


def create_video_from_frames(video_frame_paths_list, queue_id, num_frames_per_second):
    print("Creating video from", len(video_frame_paths_list), "with", num_frames_per_second, "frames per second")
    video_path = os.path.join(OUTPUT_PATH, queue_id + '/video.mp4')
    frame_input_list_file = os.path.join(OUTPUT_PATH, queue_id + '/input.txt')

    with open(frame_input_list_file, 'w') as f:
        for video_frame_path in video_frame_paths_list:
            f.write("file '" + video_frame_path + "'\n")

    print("Creating video at", video_path)
    ffmpeg_command = ['ffmpeg', '-y', '-r', str(num_frames_per_second), '-f', 'concat', '-safe', '0', '-i',
                      frame_input_list_file, '-c:v', 'libx264', '-profile:v', 'high', '-crf', '20',
                      '-pix_fmt', 'yuv420p', video_path]
    subprocess.run(ffmpeg_command)
    print("COMPLETED:", len(video_frame_paths_list), "with", num_frames_per_second, "fps -> ", video_path)
    return video_path


def exit_signal_handler(self, sig):
    # Shutdown the server gracefully when Docker requests it
    sys.stderr.write('\nBackend-Server: Shutting down...\n')
    sys.stderr.flush()
    quit()


if __name__ == "__main__":
    # Set up the exit signal handler
    signal.signal(signal.SIGTERM, exit_signal_handler)
    signal.signal(signal.SIGINT, exit_signal_handler)

    # Set up the server
    print('------------------------------------------')
    print('Starting backend server, please wait...')
    print('------------------------------------------\n\n')
    global_device, global_model, global_wm_encoder = setup()

    httpd = HTTPServer(('0.0.0.0', PORT), SimpleHTTPRequestHandler)
    print('------------------------------------------')
    print('Backend Server ready for processing on port', PORT)
    print('------------------------------------------')
    if not WATERMARK_FLAG:
        print('Note: Watermarking is disabled')
    else:
        print('Note: Watermarking is enabled with "Stable Diffusion" text')

    if not SAFETY_FLAG:
        print('Note: Safety checks are disabled - take responsibility for your own actions!')
    else:
        print(
            'Note: Safety checks are are enabled. A NSFW-replacement image will be used if an image is deemed naughty')

    if IMAGE_QUALITY == "MAX":
        print("Note: Image output will be at maximum quality with PNG format - 420-500 Kb per image, upscale 4MB-5MB")
    elif IMAGE_QUALITY == "HIGH":
        print("Note: Image output will be in high quality with JPG format - 120-200 Kb per image, upscale 1-2 MB")
    elif IMAGE_QUALITY == "LOW":
        print("Note: Image output will be of lower quality with JPG format - 15-30 Kb per image, upscale 400-800 Kb")

    httpd.serve_forever()
