# FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
# expose
EXPOSE 8000

# set working directory
WORKDIR /app

# install pip
RUN apt-get update && apt-get install -y python3-pip

# install git
RUN apt-get install -y git python3-setuptools curl ffmpeg libsm6 libxext6

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'export PATH="$HOME/.cargo/env:$PATH"' >> $HOME/.bashrc

# update pip
RUN pip3 install --upgrade pip

# install jax[cuda]
# RUN pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip3 install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install Jupyter so this build can be set running on its own if necessary
# RUN pip3 install --upgrade jupyter
# install python packageswe know works

RUN pip3 install --upgrade cython
RUN pip3 install --upgrade setuptools-rust
RUN pip3 install --upgrade matplotlib
RUN pip3 install --upgrade opencv-python==4.5.4.60

RUN pip3 install --upgrade more_itertools~=8.12.0
# RUN pip3 install --upgrade youtokentome~=1.0.6
RUN pip3 install --upgrade omegaconf>=2.0.0
RUN pip3 install --upgrade einops~=0.3.2
RUN pip3 install --upgrade segmentation-models-pytorch==0.1.3

RUN pip3 install --upgrade PyWavelets==1.1.1
RUN PATH="$HOME/.cargo/bin:$PATH" && pip3 install --upgrade transformers~=4.19.2  # was 4.10.2

# install more requirements!
RUN pip3 install --upgrade albumentations==0.4.3
RUN pip3 install --upgrade diffusers
RUN pip3 install --upgrade pudb==2019.2
RUN pip3 install --upgrade invisible-watermark
RUN pip3 install --upgrade imageio==2.9.0
RUN pip3 install --upgrade imageio-ffmpeg==0.4.2
RUN pip3 install --upgrade pytorch-lightning==1.6.0
RUN pip3 install --upgrade omegaconf==2.1.1
RUN pip3 install --upgrade test-tube>=0.7.5
RUN pip3 install --upgrade streamlit>=0.73.1
RUN pip3 install --upgrade einops==0.3.0
RUN pip3 install --upgrade torch-fidelity==0.3.0
RUN pip3 install --upgrade kornia==0.6
RUN pip3 install --upgrade torchmetrics==0.6.0

# Finally a couple more requirements
RUN pip3 install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
RUN pip3 install -e git+https://github.com/openai/CLIP.git@main#egg=clip


# add source code
COPY . /app

EXPOSE 8000
# Start Jupyter notebook server
CMD ["python3", "-u", "server.py"]
#CMD  python -m http.server 8000

# CMD python3 scripts/txt2img.py --prompt "sexy twenty year old guy with stubble, photograph" --plms --outdir "/library" --n_iter 50 --ddim_steps 50 --seed 1002


