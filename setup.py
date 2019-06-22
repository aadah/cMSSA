from setuptools import setup, find_packages
from pip._internal.req import parse_requirements

g = parse_requirements('requirements.txt', session='')
install_requires = [str(req.req) for req in g]

setup(
   name='cMSSA',
   version='1.0.0',
   description='Contrastive Multivariate Singular Spectrum Analysis',
   author='Abdi-Hakin Dirie',
   packages=find_packages(),
   install_requires=install_requires,
)
