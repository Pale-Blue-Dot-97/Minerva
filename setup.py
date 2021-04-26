from setuptools import setup
from os import path

# Read contents of README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='Minerva',
      version='0.1',
      description='Package to build, train and test neural network models on land cover data',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Harry James Baker',
      author_email='hjb1d20@soton.ac.uk',
      url='https://github.com/Pale-Blue-Dot-97/Minerva',
      license='GNU GPLv3',
      license_files='LICENCE.txt',
      packages=['Minerva',
                'Minerva.utils'],
      scripts=['Minerva/bin/DownloadStrapper.py',
               'Minerva/bin/Landcovernet_Download_API.py',
               'Minerva/bin/MinervaPercep.py'],
      package_data={'landcovernet': ['data/landcovernet']}
      )
