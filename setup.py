from setuptools import setup

setup(name='Minerva',
      version='0.1',
      description='Package to build, train and test neural network models on land cover data',
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
