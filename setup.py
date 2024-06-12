# =============================================================================#
# Authors: Windsor Nguyen
# File: setup.py
# =============================================================================#

"""Spectral State Space Models."""

import setuptools


setuptools.setup(
    name='spectral_ssm',
    version='1.0',
    description='Dependency manager for Google DeepMind Spectral State Space model',
    long_description="""
        Spectral State Space Models. See more details in the
        [`README.md`](https://github.com/windsornguyen/spectral_ssm).
        """,
    long_description_content_type='text/markdown',
    author='Yagiz Devre, Evan Dogariu, Chiara von Gerlach, Isabel Liu, Windsor Nguyen, Dwaipayan Saha, Daniel Suo',
    author_email='mn4560@princeton.edu',
    url='https://github.com/windsornguyen/spectral_ssm',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=2.0',
    ],
    python_requires='>=3.9, <3.12',
    extras_require={'dev': ['ipykernel>=6.29.4', 'ruff>=0.4.8']},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='pytorch machine learning spectral state space models',
)
