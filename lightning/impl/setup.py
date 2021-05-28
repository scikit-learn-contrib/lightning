import os.path

from numpy import get_include
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    config = Configuration('impl', parent_package, top_path)

    randomdir = os.path.join(top_path, "lightning", "impl", "randomkit")
    currdir = os.path.dirname(os.path.abspath(__file__))

    files = [
        'adagrad_fast',
        'dataset_fast',
        'dual_cd_fast',
        'loss_fast',
        'prank_fast',
        'primal_cd_fast',
        'prox_fast',
        'sag_fast',
        'sdca_fast',
        'sgd_fast',
        'svrg_fast',
    ]
    for f in files:
        config.add_extension(f,
                             sources=[f'{f}.pyx'],
                             language='c++',
                             include_dirs=[get_include(), randomdir])

        # add .pxd files to be re-used by third party software
        pxd_file = os.path.join(currdir, f'{f}.pxd')
        if os.path.exists(pxd_file):
            config.add_data_files(f'{f}.pxd')

    config.add_subpackage('datasets')
    config.add_subpackage('randomkit')
    config.add_subpackage('tests')

    return config

if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
