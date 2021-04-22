import os

if __name__ == '__main__':
    os.system('python Landcovernet_Download_API.py --classes Water Artificial_Bareground Permanent_Snow/Ice ' +
              '--items 50 50 50')
