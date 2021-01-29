from setuptools import setup
from setuptools import find_packages

packages_list = [f"headlines_project.{s}" for s in find_packages(where="headlines_project")]

setup(name='headlines_project',
      version='0.0.1',
      description='Virality in headlines_project headlines_project',
      packages=packages_list,
      zip_safe=False)