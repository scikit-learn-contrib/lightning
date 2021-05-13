import sys

from numpy import get_include
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    rnd_kit = 'randomkit'
    rnd_fast = 'random_fast'
    config = Configuration(rnd_kit, parent_package, top_path)
    libs = []
    if sys.platform == 'win32':
        libs.append('Advapi32')

    config.add_extension(
        rnd_fast,
        sources=[f'{rnd_fast}.pyx', f'{rnd_kit}.c'],
        language='c++',
        libraries=libs,
        include_dirs=[get_include()]
    )

    config.add_subpackage('tests')
    config.add_data_files(f'{rnd_fast}.pxd')
    config.add_data_files(f'{rnd_kit}.h')

    return config

if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
