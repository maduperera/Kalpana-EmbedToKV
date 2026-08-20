from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kalpana-embed-to-kv",
    version="1.0.0",
    author="Vijñāna AI",
    author_email="support@vijnanaai.com",
    description="O(1) Holographic Memory KV Cache Replacement Engine powered by Resonant Interference Fields",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maduperera/Kalpana-EmbedToKV",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
    ],
    extras_require={
        "transformers": ["transformers>=4.30.0", "sentence-transformers>=2.2.0"],
        "dev": ["pytest>=7.0.0"],
    },
)
