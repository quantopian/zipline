from itertools import product
import os

py_versions = ('2.7', '3.4')
numpy_versions = ('1.9', '1.10')


def main():
    for pair in product(py_versions, numpy_versions):
        os.system('conda build conda/zipline -c quantopian '
                  '--python=%s --numpy=%s' % pair)


if __name__ == '__main__':
    main()
