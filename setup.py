# Import needed function from setuptools
from setuptools import setup

# Create proper setup to be used by pip
setup(name='ff_functions',
      version='0.0.1',
      author='Mark',
      packages=['ff_functions'],
      install_requires=['pandas'])


      import sqlite3
    import os
    import datetime as dt
    from shutil import copyfile