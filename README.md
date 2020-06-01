
# gqnlib

Generative Query Network by PyTorch

# Requirements

* Python == 3.7
* PyTorch == 1.5.0

Requirements for example code

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
pip3 install tqdm==4.46.0 tensorflow==2.2.0 tensorboardX==2.0
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

Dataset is provided by DeepMind, and you can see how to download them from [DeepMind GitHub](https://github.com/deepmind/gqn-datasets).

The following command will download the specified dataset and convert tfrecords into torch gziped files. This shell script uses [`gsutil`](https://cloud.google.com/storage/docs/gsutil) command, which should be installed in advance ([read here](https://cloud.google.com/storage/docs/gsutil_install)).

**Caution**: This process takes a very long time. For example, `shepard_metzler_5_parts` dataset which is the smallest one takes 2~3 hours on my PC with 32 GB memory.

**Caution**: This process creates very large size files. For example, original `shepard_metzler_5_parts` dataset contains 900 files (17 GB) for train and 100 files (5 GB) for test, and converted dataset contains 2,100 files (47 GB) for train and 400 files (12 GB) for test.

```bash
bash bin/download.sh
```

## Run experiment

Shell script `bin/train.sh` contains the necessary settings.

```bash
# bin/train.sh
MODEL=gqn  # Model name (gqn, cgqn, sgqn)
DATASET=shepard_metzler_5_parts  # Dataset name
CUDA=0,1  # GPU IDs
STEPS=2000000  # Max steps
TEST_INTERVAL=200000  # Test interval (steps)
```

Run training. This takes a very long time, 10~30 hours.

```bash
bash bin/train.sh
```

# Reference

* S. M. Ali Eslami et al., "Neural scene representation and rendering," [Science Vol. 360, Issue 6394, pp.1204-1210 (15 Jun 2018)](https://science.sciencemag.org/content/360/6394/1204.full?ijkey=kGcNflzOLiIKQ&keytype=ref&siteid=sci)
* A. Kumar et al., "Consistent Generative Query Network," [arXiv](http://arxiv.org/abs/1807.02033)
* T. Ramalho et al., "Encoding Spatial Relations from Natural Language," [arXiv](http://arxiv.org/abs/1807.01670)
* DeepMind. [Blog post](https://deepmind.com/blog/article/neural-scene-representation-and-rendering)
* Datasets by DeepMind for GQN. [GitHub](https://github.com/deepmind/gqn-datasets)
* Datasetf by DeepMind for SLIM. [GitHub](https://github.com/deepmind/slim-dataset)
