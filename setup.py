from setuptools import find_packages
from setuptools import setup



with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='functions',
      version="0.0.2",
      description="new environment for testing",
      install_requires=requirements,
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True)
