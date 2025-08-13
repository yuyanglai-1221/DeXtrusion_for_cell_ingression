from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='dextrusion',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.0.8',
    description='DeXtrusion: automatic detection of cell extrusion in epithelial tissu',
      author='Gaëlle Letort and Alexis Villars',
      url='https://gitlab.pasteur.fr/gletort/dextrusion',
      package_dir={'':'src'},
      packages=find_packages('src'),
    install_requires=[
        "matplotlib",
        "numpy",
        "opencv-python",
        "tifffile>=2022.2.2",
        "roifile",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "tensorflow==2.8", 
        "protobuf==3.19",
        "ipython"
    ],
)

