from setuptools import setup, find_packages
from os import path

__version__ = '0.1.0'
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

setup(
    name='assideo',
    version=__version__,
    description='Deep metric learning tools in pytorch',
    long_description=long_description,
    # long_description_content_type='text/markdown',
    url='https://github.com/BaranowskiBrt/Assideo',
    author='Bartłomiej Baranowski',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch >= 1.9', 'torchvision', 'timm >= 0.4.12', 'omegaconf>=2.0',
        'tqdm>=4.62', 'opencv-python>=4.5', 'scikit-learn>=1.0', 'wandb>=0.12'
    ],
    python_requires='>=3.7',
)