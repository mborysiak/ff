# Import needed function from setuptools
from setuptools import setup

# Create proper setup to be used by pip
setup(name='ff_functions',
      version='0.0.1',
      author='Mark',
      packages=['ff', 'ff.modeling'],
      install_requires=['pandas', 'datetime'])