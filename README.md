
# gqnlib

Generative Query Network by PyTorch

# Requirements

* Python == 3.7
* PyTorch == 1.5.0

Requirements for example code

* matplotlib == 3.2.1
* tqdm == 4.46.0
* tensorflow == 2.2.0
* tensorboardX == 2.0

# How to use

## Set up environments

Clone repository.

```bash
git clone https://github.com/rnagumo/gqnlib.git
cd gqnlib
```

Install the package in virtual env.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install .

# Install other requirements for sample code.
pip3 install matplotlib==3.2.1 tqdm==4.46.0 tensorflow==2.2.0 tensorboardX==2.0
```

Or use [Docker](https://docs.docker.com/get-docker/) and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

```bash
docker build -t dgmvae .
docker run -it dgmvae bash
```

You can run container with GPUs by Docker 19.03.

```bash
docker run --gpus all -it dgmvae bash
```

## Prepare dataset

Dataset is provided by deepmind, and you can see how to download them from [GitHub](https://github.com/deepmind/gqn-datasets).

The following commant will download the specified dataset. This shell script uses [`gsutil cp`]((https://cloud.google.com/storage/docs/gsutil/commands/cp)) command.

```bash
bash bin/download.sh shepard_metzler_5_parts
```

## Run experiment

Train models. Shell scripts in `bin` folder contains the necessary settings of the environment variables.

```bash
# Usage
bash bin/train.sh <model-name> <random-seed>

# Example
bash bin/train.sh cnp 0
```

# Reference

Original paper

* S. M. Ali Eslami et al., "Neural scene representation and rendering," [Science Vol. 360, Issue 6394, pp.1204-1210 (15 Jun 2018)](https://science.sciencemag.org/content/360/6394/1204.full?ijkey=kGcNflzOLiIKQ&keytype=ref&siteid=sci)
* DeepMind, [Blog post](https://deepmind.com/blog/article/neural-scene-representation-and-rendering)
* Datasets by DeepMind, [GitHub](https://github.com/deepmind/gqn-datasets)

Reference implementations

* wohlert [GitHub](https://github.com/wohlert/generative-query-network-pytorch)
* iShohei220 [GitHub](https://github.com/iShohei220/torch-gqn)
