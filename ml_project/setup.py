from setuptools import find_packages, setup

setup(
    name="ml_pipeline",
    packages=find_packages(),
    version="0.1.0",
    description="Homework1",
    author="Anna Goremykina",
    install_requires=[
        "click==7.1.2",
        "python-dotenv>=0.17.0",
        "scikit-learn==0.24.1",
        "dataclasses==0.8",
        "pyyaml==3.11",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.1.5",
        "numpy==1.19.5"
    ],
    license="MIT",
)
