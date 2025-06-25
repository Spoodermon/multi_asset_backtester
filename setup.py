from setuptools import setup, find_packages

setup(
    name="multi_asset_backtester",
    version="0.1.0",
    description="A Python package for multi-asset backtesting.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas"
    ],
    include_package_data=True,
    python_requires=">=3.7",
)