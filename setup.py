from setuptools import setup, find_packages

setup(
    name="cryosiam",
    version="0.1.0",
    author="Frosina Stojanovska",
    author_email="stojanovska.frose@gmail.com",
    description="CryoSiam: Deep Learning-based CryoET Analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/frosinastojanovska/cryosiam",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "cryosiam = cryosiam.cli:main"
        ]
    },
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "mrcfile",  # Example dependencies
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
