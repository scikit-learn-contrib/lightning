from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

from lightning._build_utils import maybe_cythonize_extensions


def configuration(parent_package='', top_path=None):
    config = Configuration('lightning', parent_package, top_path)

    config.add_subpackage('impl')

    maybe_cythonize_extensions(top_path, config)

    return config

if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
