
from setuptools import setup, find_packages


install_requires = [
    "torch==1.5.0",
    "torchvision==0.6.0",
    "tqdm==4.46.0",
    "tensorflow==2.2.0",
    "tensorboardX==2.0",
    "matplotlib==3.2.1",
]


setup(
    name="gqnlib",
    version="0.1",
    description="Generative Query Network by PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
)
