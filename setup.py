from setuptools import setup
from setuptools import find_packages

packages_list = [f"headlines_project.{s}" for s in find_packages(where="headlines_project")]

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='headlines_project',
      version='0.0.2',
      install_requires=required,
      description='Virality in headlines_project headlines_project',
      packages=packages_list,
      zip_safe=False)