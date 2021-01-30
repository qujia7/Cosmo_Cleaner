from distutils.core import setup, Extension
import os



setup(name='cosmo_cleaner',
      version='0.1',
      description='Redshift cleaning',
      author='Omar Darwish,Frank Qu',
      license='BSD-2-Clause',
      packages=['cosmo_cleaner'],
      package_dir={'cosmo_cleaner':'cosmo_cleaner'},
      zip_safe=False)