#! /usr/bin/env python
#
# Copyright (C) 2012 Mathieu Blondel

import re
import sys
import os

from distutils.command.sdist import sdist

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


DISTNAME = 'sklearn-contrib-lightning'
DESCRIPTION = ("Large-scale sparse linear classification, "
               "regression and ranking in Python")
with open('README.rst', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Mathieu Blondel'
MAINTAINER_EMAIL = 'mathieu@mblondel.org'
URL = 'https://github.com/scikit-learn-contrib/lightning'
LICENSE = 'new BSD'
with open(os.path.join('lightning', '__init__.py'), encoding='utf-8') as f:
    match = re.search(r'__version__[ ]*=[ ]*[\"\'](?P<version>.+)[\"\']',
                      f.read())
    VERSION = match.group('version').strip()
MIN_PYTHON_VERSION = '3.6'
with open('requirements.txt', encoding='utf-8') as f:
    REQUIREMENTS = [
        line.strip()
        for line in f.read().splitlines()
        if line.strip()
    ]


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('lightning')

    return config

if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          python_requires=f'>={MIN_PYTHON_VERSION}',
          install_requires=REQUIREMENTS,
          include_package_data=True,
          scripts=["bin/lightning_train",
                   "bin/lightning_predict"],
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          cmdclass={"sdist": sdist},
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python :: 3',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
             ]
          )
