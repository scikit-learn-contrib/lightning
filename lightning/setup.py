import sys

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


def cythonize_extensions(top_path, config):
    try:
        from Cython.Build import cythonize
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            'Please install Cython in order to build a lightning from source.') from e

    config.ext_modules = cythonize(config.ext_modules,
                                   compiler_directives={'language_level': 3})


def configuration(parent_package='', top_path=None):
    config = Configuration('lightning', parent_package, top_path)

    config.add_subpackage('impl')

    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if 'sdist' not in sys.argv:
        cythonize_extensions(top_path, config)

    return config

if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
