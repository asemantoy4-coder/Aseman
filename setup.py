# fast-scalp-crypto-bot/setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fast-scalp-crypto-bot",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fast Scalp Trading Bot for Crypto Markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asemantoy4-coder/fast-scalp-crypto-bot",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ccxt>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-telegram-bot>=20.0",
        "ta>=0.10.0",
        "python-dotenv>=1.0.0",
        "schedule>=1.2.0",
        "aiohttp>=3.8.0",
    ],
    entry_points={
        "console_scripts": [
            "fast-scalp=main:main",
        ],
    },
)
