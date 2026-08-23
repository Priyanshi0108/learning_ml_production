from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Read requirements.txt and return a list of packages.
    """

    with open(file_path, "r") as f:
        requirements = f.readlines()

    requirements = [req.strip() for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="mlProject",
    version="0.0.1",
    author="Priyanshi",
    author_email="singhamanrajpoot2007@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)