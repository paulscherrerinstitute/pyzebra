import re

from setuptools import find_packages, setup

with open("pyzebra/__init__.py") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

setup(
    name="pyzebra",
    version=version,
    description="An experimental data analysis library for zebra instrument.",
    packages=find_packages(),
    license="GNU GPLv3",
)
