from __future__ import print_function

import os
from shutil import copyfile

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

# full URLS:
# https://drive.google.com/uc?export=download&id=1aklDTwEYcZ0AdFpjWkUx2fupzNqI33FH # test
# https://drive.google.com/uc?export=download&id=1HHhdAXEvvgPh9ybUr80RpqRfZOEqVC-i" # train

URL = ['https://drive.google.com/uc?export=download&id=1HHhdAXEvvgPh9ybUr80RpqRfZOEqVC-i',
        'https://drive.google.com/uc?export=download&id=1aklDTwEYcZ0AdFpjWkUx2fupzNqI33FH']
DATA = ['train.csv.zip', 'test.csv.zip']

def main(output_dir='data'):
    filenames = DATA
    full_urls = URL
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for url, filename in zip(full_urls, filenames):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            continue

        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))

if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()