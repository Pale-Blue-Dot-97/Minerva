from setuptools import setup

setup(name='Minerva',
      version='0.1',
      packages=['Minerva',
                'Minerva.utils'],
      scripts=['Minerva/bin/DownloadStrapper.py',
               'Minerva/bin/Landcovernet_Download_API.py',
               'Minerva/bin/MinervaPercep.py'],
      package_data={'landcovernet': ['data/landcovernet']}
      )
