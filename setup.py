import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="state_smoother",
    version="0.1.0",
    author="Jason M. Cherry",
    author_email="jcherry@gmail.com",
    description="A custom implementation of exponential smoothing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Calvinxc1/StateSmoother",
    packages=setuptools.find_packages(),
    install_requires=[
        'matplotlib >= 3.1, < 4',
        'numpy >= 1.18, < 2',
        'scipy >= 1.4, < 2',
        'seaborn >= 0.1, < 1',
        'torch >= 1.4, < 2',
        'tqdm >= 4.45, < 5',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
    ],
    python_requires='>= 3.6',
)