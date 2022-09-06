import os
import random
import urllib.request
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image, ImageDraw
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

import uuid
import json
import sys
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote

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
MODEL_PATH = 'models/ldm/stable-diffusion-v1/model.ckpt'
SCALE = 7.5  # was opt.scale#
DDIM_STEPS = 50  # was opt.ddim_steps (number of ddim sampling steps)
DDIM_ETA = 0.0  # was opt.ddim_eta  (ddim eta (eta=0.0 corresponds to deterministic sampling)
N_SAMPLES = 1  # was opt.n_samples (how many samples to produce for each given prompt. A.k.a. batch size)
PRECISION = "autocast"  # can be "autocast" or "full"
STRENGTH = 0.75  # was opt.strength - used when processing an image - 0 means no change through 0.999 means full change
OUTPUT_PATH = '/library'
PORT = 8080

# GLOBAL VARS
global_device = None
global_model = None
global_wm_encoder = None


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


def load_and_format_image(path):
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

    # make sure both width and height of the image are <= 512:
    print(f"loaded input image from {path}")

    w, h = image.size
    print(f"image size ({w}, {h})")

    old_size = image.size  # old_size[0] is in (width, height) format

    ratio = float(512) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    image.thumbnail(new_size, Image.Resampling.LANCZOS)
    resized_image = Image.new("RGB", (512, 512))
    resized_image.paste(image, ((512 - new_size[0]) // 2,
                            (512 - new_size[1]) // 2))

    w, h = resized_image.size
    print(f"Image resized to size ({w}, {h}) ")

    image = np.array(resized_image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    processed_image = 2. * image - 1.

    # Commented out because burning the words "Original Image" into the image was disrupting the 'advanced' workflow
    # where I take an image form the libray to manipulate it further. You should know the original image!
    # ImageDraw.Draw(resized_image).text((10, 10), "Original Image", fill=(255, 255, 255))
    return resized_image, processed_image


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def orig_check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def check_safety(x_image):
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


def process(text_prompt, device, model, wm_encoder, queue_id, num_images, options):
    print('Running Prompt Processing')
    sampler = PLMSSampler(model)  # Uses PLMS model
    seed_everything(options['seed'])
    start = time.time()
    library_dir_name = os.path.join(OUTPUT_PATH, queue_id)
    os.makedirs(library_dir_name, exist_ok=True)
    base_count = len(os.listdir(library_dir_name))

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
                            uc = None
                            if SCALE != 1.0:
                                uc = model.get_learned_conditioning(N_SAMPLES * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [LATENT_CHANNELS, options['height'] // options['downsampling_factor'],
                                     options['width'] // options['downsampling_factor']]
                            samples_ddim, _ = sampler.sample(S=options['ddim_steps'],
                                                             conditioning=c,
                                                             batch_size=N_SAMPLES,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=options['scale'],
                                                             unconditional_conditioning=uc,
                                                             eta=options['ddim_eta'],
                                                             x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            # REPLACE WITH orig_check_safety() to re-enable the safety check
                            # x_checked_image, has_nsfw_concept = orig_check_safety(x_samples_ddim)
                            x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(library_dir_name, f"{str(uuid.uuid4())}.png"))
                                base_count += 1

                            end = time.time()
                            time_taken = end - start

                            save_metadata_file(base_count, library_dir_name, options, queue_id, text_prompt, time_taken,
                                               '', '')

        return {'success': True, 'queue_id': queue_id}

    except Exception as e:
        print(e)
        end = time.time()
        time_taken = end - start
        save_metadata_file(base_count, library_dir_name, options, queue_id, text_prompt, time_taken, str(e), '')
        return {'success': False, 'error: ': 'error: ' + str(e), 'queue_id': queue_id}


def process_image(original_image_path, text_prompt, device, model, wm_encoder, queue_id, num_images, options):
    print('Running Image Processing')
    sampler = DDIMSampler(model)  # Uses DDIM model
    seed_everything(options['seed'])
    start = time.time()
    library_dir_name = os.path.join(OUTPUT_PATH, queue_id)
    os.makedirs(library_dir_name, exist_ok=True)
    base_count = len(os.listdir(library_dir_name))

    try:
        assert text_prompt is not None
        data = [N_SAMPLES * [text_prompt]]

        # load the image
        resized_image, init_image = load_and_format_image(original_image_path)
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=N_SAMPLES)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=options['ddim_steps'], ddim_eta=DDIM_ETA, verbose=False)

        assert 0. <= options['strength'] <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(options['strength'] * options['ddim_steps'])
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast if PRECISION == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for n in trange(int(num_images / N_SAMPLES), desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if SCALE != 1.0:
                                uc = model.get_learned_conditioning(N_SAMPLES * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * N_SAMPLES).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=options['scale'],
                                                     unconditional_conditioning=uc, )

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            # save the newly created images
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(library_dir_name, f"{base_count+1:02d}-{str(uuid.uuid4())[:8]}.png"))
                                base_count += 1

                            # save the resized original image
                            resized_image.save(os.path.join(library_dir_name, '00-original.png'))

                            end = time.time()
                            time_taken = end - start

                            save_metadata_file(base_count, library_dir_name, options, queue_id, text_prompt, time_taken,
                                               '', original_image_path)

            return {'success': True, 'queue_id': queue_id}

    except Exception as e:
        print(e)
        end = time.time()
        time_taken = end - start
        save_metadata_file(base_count, library_dir_name, options, queue_id, text_prompt, time_taken, str(e), original_image_path)
        return {'success': False, 'error: ': 'error: ' + str(e), 'queue_id': queue_id}


def save_metadata_file(base_count, library_dir_name, options, queue_id, text_prompt, time_taken, error,
                       original_image_path):
    with open(library_dir_name + '/index.json', 'w', encoding="utf8") as outfile:
        metadata = {
            "text_prompt": text_prompt,
            "num_images": base_count,
            "queue_id": queue_id,
            "time_taken": round(time_taken, 2),
            "seed": options['seed'],
            "height": options['height'],
            "width": options['width'],
            "ddim_steps": options['ddim_steps'],
            "ddim_eta": options['ddim_eta'],
            "scale": options['scale'],
            "downsampling_factor": options['downsampling_factor'],
            "error": error,
            "original_image_path": original_image_path,
            "strength": options['strength'] if "strength" in options else -1 # -1 means not applicable
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


        # Set up defaults for the optional extra properties which will
        # be overwritten should they be in the request
        options = {
            'seed': random.randint(1, 2147483647),
            'height': HEIGHT,
            'width': WIDTH,
            'ddim_steps': DDIM_STEPS,
            'ddim_eta': DDIM_ETA,
            'scale': SCALE,
            'downsampling_factor': DOWNSAMPLING_FACTOR,
            'strength': STRENGTH
        }

        # Override the default options with any in the request:
        try:
            if 'seed' in data and len(str(data['seed']).strip()) > 0 and int(data['seed']) > 0:
                options['seed'] = int(data['seed'])
            # else use the seed generated when options was initialised

        except ValueError:
            # use the seed generated when options was initialised
            pass


        if 'height' in data:
            options['height'] = int(data['height'])
        if 'width' in data:
            options['width'] = int(data['width'])
        if 'ddim_steps' in data:
            options['ddim_steps'] = int(data['ddim_steps'])
        if 'ddim_eta' in data:
            options['ddim_eta'] = float(data['ddim_eta'])
        if 'scale' in data:
            options['scale'] = float(data['scale'])
        if 'downsampling_factor' in data:
            options['downsampling_factor'] = int(data['downsampling_factor'])
        if 'strength' in data:
            options['strength'] = float(data['strength'])
            if options['strength'] >= 1.0:
                options['strength'] = 0.999
            elif options['strength'] < 0.0:
                options['strength'] = 0.001

        original_image_path = ''
        if 'original_image_path' in data:
            original_image_path = data['original_image_path'].strip()

            # only image paths which are wither URLs or are sourced from the library are allowed.
            if not (original_image_path.startswith('http') or original_image_path.startswith('library/')):
                if original_image_path != '':
                    print('Warning: "{}" is not a valid URL - processing continues as if no file present'.format(original_image_path))
                original_image_path = ''
        # process!
        if original_image_path != '':
            result = process_image(original_image_path, prompt, global_device, global_model,
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
    httpd.serve_forever()
