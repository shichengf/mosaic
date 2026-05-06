"""setup.py — install MOSAIC as an editable Python package.

Usage:
    pip install -e .          # core (CPU/GPU agnostic)
    pip install -e .[full]    # with MD tools, NetCDF, plotting extras
"""
from setuptools import setup, find_packages

CORE = [
    "numpy>=1.24,<3.0",
    "scipy>=1.10",
    "scikit-learn>=1.2",
    "torch>=2.0",
    "pytorch-lightning>=2.0",
    "matplotlib>=3.6",
    "pandas>=1.5",
    "tqdm>=4.60",
]

# Extras for data preparation on real domains.
DATA_PREP = [
    "mdtraj>=1.9",      # RNA molecular dynamics
    "biopython>=1.80",
    "xarray>=2023.1",   # ENSO climate (NetCDF)
    "netCDF4>=1.6",
    "h5py>=3.7",
    "jax>=0.4.0",       # synthetic Langevin generator
    "jaxlib>=0.4.0",
]

# Extras for plotting/UMAP/seaborn analyses in scripts/eval.
ANALYSIS = [
    "seaborn>=0.12",
    "umap-learn>=0.5",
]

setup(
    name="mosaic-crl",
    version="0.1.0",
    description=(
        "MOSAIC: Module discovery via Sparse Additive Identifiable Causal "
        "learning for scientific time series."
    ),
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(include=["mosaic", "mosaic.*"]),
    install_requires=CORE,
    extras_require={
        "data": DATA_PREP,
        "analysis": ANALYSIS,
        "full": CORE + DATA_PREP + ANALYSIS,
    },
)
