# setup.py - نسخه ساده
from setuptools import setup, find_packages

setup(
    name="crypto-trading-system",
    version="7.4.0",
    author="Crypto AI Trading System",
    description="سیستم تحلیل معاملاتی ارز دیجیتال",
    packages=find_packages(),
    install_requires=[
        'fastapi>=0.104.1',
        'uvicorn[standard]>=0.24.0',
        'requests>=2.31.0',
        'pydantic>=2.5.0',
        'pandas>=2.1.3',
        'numpy>=1.26.2',
    ],
    python_requires='>=3.9',
)