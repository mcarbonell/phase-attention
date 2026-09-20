from setuptools import setup, find_packages

setup(
    name="phase-attention",
    version="0.1.0",
    description="Multiplier-Free, Exact O(N) Linear Attention on the Unit Circle U(1) for Ultra-Low Power Edge AI",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Mario Carbonell",
    author_email="marioraulcarbonell@gmail.com",
    url="https://github.com/mcarbonell/phase-attention",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
