FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# set working directory
WORKDIR /app

# install python package dependencies using Ubunt's APT
RUN apt-get update && apt-get install -y python3-pip
RUN apt-get install -y git python3-setuptools curl ffmpeg libsm6 libxext6

# Get and install Rust - we'll be using this later for compiling some code
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'export PATH="$HOME/.cargo/env:$PATH"' >> $HOME/.bashrc

# update pip
RUN pip3 install --upgrade pip

# install jax[cuda]
RUN pip3 install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# install python packages
RUN pip3 install --upgrade cython
RUN pip3 install --upgrade setuptools-rust
RUN pip3 install --upgrade matplotlib
RUN pip3 install --upgrade opencv-python==4.5.4.60
RUN pip3 install --upgrade omegaconf>=2.0.0
RUN pip3 install --upgrade einops~=0.3.2
RUN pip3 install --upgrade segmentation-models-pytorch==0.1.3
RUN pip3 install --upgrade PyWavelets==1.1.1

# Rust will compile the transformers library from source
RUN PATH="$HOME/.cargo/bin:$PATH" && pip3 install --upgrade transformers~=4.19.2  # was 4.10.2

# install specific requirements for this project
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

# Finally a couple more requirements - getting the latest versions of these repos from Github
RUN pip3 install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
RUN pip3 install -e git+https://github.com/openai/CLIP.git@main#egg=clip


# add our backend operations app
COPY . /app

# and let's get this backend on the air!
CMD ["python3", "-u", "server.py"]


