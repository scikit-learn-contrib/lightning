import sys
import numpy

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('randomkit', parent_package, top_path)
    libs = []
    if sys.platform == 'win32':
        libs.append('Advapi32')

    config.add_extension(
        'random_fast',
        sources=['random_fast.pyx', 'randomkit.c'],
        language='c++',
        libraries=libs,
        include_dirs=[numpy.get_include()]
    )

    config.add_subpackage('tests')
    config.add_data_files('random_fast.pxd')
    config.add_data_files('randomkit.h')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
