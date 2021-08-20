from pathlib import Path

from setuptools import setup


BASE_DIR = Path(__file__).parent


with open(BASE_DIR / "requirements.txt") as file:
    required_packages = [ln.strip() for ln in file.readlines()]


test_packages = [
    "pytest==6.1.1",
    "pytest-cov==2.12.0",
    "sh==1.14.2",
]


dev_packages = [
    "awslambdaric==1.2.0",
    "fastapi==0.66.0",
    "h5py==3.3.0",
    "hydra-core==1.1.0",
    "matplotlib==3.4.2",
    "streamlit==0.83.0",
    "streamlit-drawable-canvas==0.8.0",
    "wandb==0.10.33",
    "uvicorn==0.14.0",
    "black==21.5b1",
    "flake8==3.9.2",
    "isort==5.8.0",
    "pre-commit==2.13.0",
]


setup(
    name="handwritten-multi-digit-number-recognition",
    version="0.1.0",
    license="MIT",
    description="Recognize handwritten multi-digit numbers.",
    author="King Yiu Suen",
    author_email="kingyiusuen@gmail.com",
    url="https://github.com/kingyiusuen/handwritten-multi-digit-number-recognition/",
    keywords=[
        "machine-learning",
        "deep-learning",
        "artificial-intelligence",
        "neural-network",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=[required_packages],
    extras_require={
        "dev": test_packages + dev_packages,
        "test": test_packages,
    },
)
