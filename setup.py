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
    "black==21.5b1",
    "flake8==3.9.2",
    "isort==5.8.0",
    "mypy==0.812",
    "pre-commit==2.13.0",
    "pydocstyle==6.1.1",
]

docs_packages = [
    "mkdocs==1.2.1",
    "mkdocs-macros-plugin==0.5.12",
    "mkdocs-material==7.1.9",
    "mkdocstrings==0.15.2",
]

setup(
    name="handwritten-digit-string-recognition",
    version="0.1.0",
    license="MIT",
    description="Recognize handwritten digit strings.",
    author="King Yiu Suen",
    author_email="kingyiusuen@gmail.com",
    url="https://github.com/kingyiusuen/handwritten-digit-string-recognition/",
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
        "dev": test_packages + dev_packages + docs_packages,
        "docs": docs_packages,
        "test": test_packages,
    },
)
