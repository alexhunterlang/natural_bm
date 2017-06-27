from setuptools import setup
from setuptools import find_packages


setup(name='Natural_BM',
      version='0.0.0',
      description='Natural Boltzmann Machines',
      author='Alex H. Lang',
      author_email='alexhunterlang+natural_bm@gmail.com',
      license='MIT',
      install_requires=['theano', 'scikit-image'],
      extras_require={
          'h5py': ['h5py'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-cov'],
      },
      packages=find_packages())
