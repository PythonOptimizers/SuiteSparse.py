#!/usr/bin/env python

########################################################################################################################
#                                                                                                                      #
#                              The file `setup.py` is automatically generated from `config/setup.cpy`                  #
#                                                                                                                      #
########################################################################################################################

from config.version import find_version, read
from config.config import get_path_option

from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension

import numpy as np

import configparser
import os
import copy

from codecs import open
from os import path

###################################################################s####################################################
# HELPERS
########################################################################################################################
def prepare_Cython_extensions_as_C_extensions(extensions):
    """
    Modify the list of sources to transform `Cython` extensions into `C` extensions.

    Args:
        extensions: A list of (`Cython`) `distutils` extensions.

    Warning:
        The extensions are changed in place. This function is not compatible with `C++` code.

    Note:
        Only `Cython` source files are modified into their `C` equivalent source files. Other file types are unchanged.

    """
    for extension in extensions:
        c_sources = list()
        for source_path in extension.sources:
            path, source = os.path.split(source_path)
            filename, ext = os.path.splitext(source)

            if ext == '.pyx':
                c_sources.append(os.path.join(path, filename + '.c'))
            elif ext in ['.pxd', '.pxi']:
                pass
            else:
                # copy source as is
                c_sources.append(source_path)

        # modify extension in place
        extension.sources = c_sources

###################################################################s####################################################
# INIT
########################################################################################################################
suitesparse_config_file = 'suitesparse.cfg'
suitesparse_config = configparser.SafeConfigParser()
suitesparse_config.read(suitesparse_config_file)

numpy_include = np.get_include()

# Use Cython?
use_cython = suitesparse_config.getboolean('CODE_GENERATION', 'use_cython')
if use_cython:
    try:
        from Cython.Distutils import build_ext
        from Cython.Build import cythonize
    except ImportError:
        raise ImportError("Check '%s': Cython is not properly installed." % suitesparse_config_file)

# Use CySparse?
use_cysparse = suitesparse_config.getboolean('CODE_GENERATION', 'use_cysparse')
cysparse_rootdir = None
if use_cysparse:
    cysparse_rootdir = get_path_option(suitesparse_config, 'CYSPARSE', 'cysparse_rootdir')
    if cysparse_rootdir == '':
        raise ValueError("You must specify the location of the CySparse source code in %s." % suitesparse_config_file)

# Debug mode?
use_debug_symbols = suitesparse_config.getboolean('CODE_GENERATION', 'use_debug_symbols')

# DEFAULT
default_include_dir = get_path_option(suitesparse_config, 'DEFAULT', 'include_dirs')
default_library_dir = get_path_option(suitesparse_config, 'DEFAULT', 'library_dirs')

suitesparse_include_dirs = get_path_option(suitesparse_config, 'SUITESPARSE', 'include_dirs')
if suitesparse_include_dirs == '':
    suitesparse_include_dirs = default_include_dir
suitesparse_library_dirs = get_path_option(suitesparse_config, 'SUITESPARSE', 'library_dirs')
if suitesparse_library_dirs == '':
    suitesparse_library_dirs = default_library_dir

########################################################################################################################
# EXTENSIONS
########################################################################################################################
include_dirs = [numpy_include, '.']

ext_params = {}
ext_params['include_dirs'] = include_dirs
# -Wno-unused-function is potentially dangerous... use with care!
# '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION': doesn't work with Cython... because it **does** use a deprecated version...


if not use_debug_symbols:
    ext_params['extra_compile_args'] = ["-O2", '-std=c99', '-Wno-unused-function']
    ext_params['extra_link_args'] = []
else:
    ext_params['extra_compile_args'] = ["-g", '-std=c99', '-Wno-unused-function']
    ext_params['extra_link_args'] = ["-g"]

##########################
# SuiteSparse
##########################
# Base solver
base_solver_ext_params = copy.deepcopy(ext_params)

base_ext = [

  
        #TODO: remove linalg
        Extension(name="suitesparse.solver_INT32_t_FLOAT64_t",
                  sources=['suitesparse/solver_INT32_t_FLOAT64_t.pxd',
                           'suitesparse/solver_INT32_t_FLOAT64_t.pyx'], **base_solver_ext_params),
    
        #TODO: remove linalg
        Extension(name="suitesparse.solver_INT32_t_COMPLEX128_t",
                  sources=['suitesparse/solver_INT32_t_COMPLEX128_t.pxd',
                           'suitesparse/solver_INT32_t_COMPLEX128_t.pyx'], **base_solver_ext_params),
    

  
        #TODO: remove linalg
        Extension(name="suitesparse.solver_INT64_t_FLOAT64_t",
                  sources=['suitesparse/solver_INT64_t_FLOAT64_t.pxd',
                           'suitesparse/solver_INT64_t_FLOAT64_t.pyx'], **base_solver_ext_params),
    
        #TODO: remove linalg
        Extension(name="suitesparse.solver_INT64_t_COMPLEX128_t",
                  sources=['suitesparse/solver_INT64_t_COMPLEX128_t.pxd',
                           'suitesparse/solver_INT64_t_COMPLEX128_t.pyx'], **base_solver_ext_params),
    

]

# UMFPACK
umfpack_ext_params = copy.deepcopy(ext_params)
umfpack_ext_params['include_dirs'].extend(suitesparse_include_dirs)
umfpack_ext_params['library_dirs'] = suitesparse_library_dirs
umfpack_ext_params['libraries'] = ['umfpack', 'amd']

umfpack_ext = [

  
        Extension(name="suitesparse.umfpack.umfpack_solver_base_INT32_t_FLOAT64_t",
                  sources=['suitesparse/umfpack/umfpack_solver_base_INT32_t_FLOAT64_t.pxd',
                           'suitesparse/umfpack/umfpack_solver_base_INT32_t_FLOAT64_t.pyx'], **umfpack_ext_params),
        # GENERIC VERSION
        Extension(name="suitesparse.umfpack.generic_solver.umfpack_generic_solver_INT32_t_FLOAT64_t",
                  sources=['suitesparse/umfpack/generic_solver/umfpack_generic_solver_INT32_t_FLOAT64_t.pxd',
                           'suitesparse/umfpack/generic_solver/umfpack_generic_solver_INT32_t_FLOAT64_t.pyx'], **umfpack_ext_params),
    
        Extension(name="suitesparse.umfpack.umfpack_solver_base_INT32_t_COMPLEX128_t",
                  sources=['suitesparse/umfpack/umfpack_solver_base_INT32_t_COMPLEX128_t.pxd',
                           'suitesparse/umfpack/umfpack_solver_base_INT32_t_COMPLEX128_t.pyx'], **umfpack_ext_params),
        # GENERIC VERSION
        Extension(name="suitesparse.umfpack.generic_solver.umfpack_generic_solver_INT32_t_COMPLEX128_t",
                  sources=['suitesparse/umfpack/generic_solver/umfpack_generic_solver_INT32_t_COMPLEX128_t.pxd',
                           'suitesparse/umfpack/generic_solver/umfpack_generic_solver_INT32_t_COMPLEX128_t.pyx'], **umfpack_ext_params),
    

  
        Extension(name="suitesparse.umfpack.umfpack_solver_base_INT64_t_FLOAT64_t",
                  sources=['suitesparse/umfpack/umfpack_solver_base_INT64_t_FLOAT64_t.pxd',
                           'suitesparse/umfpack/umfpack_solver_base_INT64_t_FLOAT64_t.pyx'], **umfpack_ext_params),
        # GENERIC VERSION
        Extension(name="suitesparse.umfpack.generic_solver.umfpack_generic_solver_INT64_t_FLOAT64_t",
                  sources=['suitesparse/umfpack/generic_solver/umfpack_generic_solver_INT64_t_FLOAT64_t.pxd',
                           'suitesparse/umfpack/generic_solver/umfpack_generic_solver_INT64_t_FLOAT64_t.pyx'], **umfpack_ext_params),
    
        Extension(name="suitesparse.umfpack.umfpack_solver_base_INT64_t_COMPLEX128_t",
                  sources=['suitesparse/umfpack/umfpack_solver_base_INT64_t_COMPLEX128_t.pxd',
                           'suitesparse/umfpack/umfpack_solver_base_INT64_t_COMPLEX128_t.pyx'], **umfpack_ext_params),
        # GENERIC VERSION
        Extension(name="suitesparse.umfpack.generic_solver.umfpack_generic_solver_INT64_t_COMPLEX128_t",
                  sources=['suitesparse/umfpack/generic_solver/umfpack_generic_solver_INT64_t_COMPLEX128_t.pxd',
                           'suitesparse/umfpack/generic_solver/umfpack_generic_solver_INT64_t_COMPLEX128_t.pyx'], **umfpack_ext_params),
    

    ]

if use_cysparse:
    umfpack_ext_params['include_dirs'].extend(cysparse_rootdir)


  

    umfpack_ext.append(
        Extension(name="suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT32_t_FLOAT64_t",
                  sources=['suitesparse/umfpack/cysparse_solver/umfpack_cysparse_solver_INT32_t_FLOAT64_t.pxd',
                           'suitesparse/umfpack/cysparse_solver/umfpack_cysparse_solver_INT32_t_FLOAT64_t.pyx'], **umfpack_ext_params)
        )
    

    umfpack_ext.append(
        Extension(name="suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT32_t_COMPLEX128_t",
                  sources=['suitesparse/umfpack/cysparse_solver/umfpack_cysparse_solver_INT32_t_COMPLEX128_t.pxd',
                           'suitesparse/umfpack/cysparse_solver/umfpack_cysparse_solver_INT32_t_COMPLEX128_t.pyx'], **umfpack_ext_params)
        )
    

  

    umfpack_ext.append(
        Extension(name="suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT64_t_FLOAT64_t",
                  sources=['suitesparse/umfpack/cysparse_solver/umfpack_cysparse_solver_INT64_t_FLOAT64_t.pxd',
                           'suitesparse/umfpack/cysparse_solver/umfpack_cysparse_solver_INT64_t_FLOAT64_t.pyx'], **umfpack_ext_params)
        )
    

    umfpack_ext.append(
        Extension(name="suitesparse.umfpack.cysparse_solver.umfpack_cysparse_solver_INT64_t_COMPLEX128_t",
                  sources=['suitesparse/umfpack/cysparse_solver/umfpack_cysparse_solver_INT64_t_COMPLEX128_t.pxd',
                           'suitesparse/umfpack/cysparse_solver/umfpack_cysparse_solver_INT64_t_COMPLEX128_t.pyx'], **umfpack_ext_params)
        )
    





########################################################################################################################
# config
########################################################################################################################
packages_list = ['suitesparse',
            'suitesparse.umfpack',
            'suitesparse.umfpack.generic_solver',
            'tests'
            ]

if use_cysparse:
    packages_list.append('suitesparse.umfpack.cysparse_solver')

ext_modules = base_ext + umfpack_ext

########################################################################################################################
# PACKAGE PREPARATION FOR EXCLUSIVE C EXTENSIONS
########################################################################################################################
# We only use the C files **without** Cython. In fact, Cython doesn't need to be installed.
if not use_cython:
    prepare_Cython_extensions_as_C_extensions(ext_modules)

########################################################################################################################
# PACKAGE SPECIFICATIONS
########################################################################################################################

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Cython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

here = path.abspath(path.dirname(__file__))
# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup_args = {
    'name' :  'SuiteSparse.py',
    'version' : find_version(os.path.realpath(__file__), 'suitesparse', '__init__.py'),
    'description' : 'A Cython library for sparse matrices',
    'long_description' : long_description,
    #Author details
    'author' : 'Nikolaj van Omme, Sylvain Arreckx, Dominique Orban',

    'author_email' : 'suitesparse\@TODO.com',

    'maintainer' : "SuiteSparse.py Developers",

    'maintainer_email' : "nikolaj@funartech.com",

    'summary' : "Fast sparse matrix library for Python",
    'url' : "https://github.com/PythonOptimizers/SuiteSparse.py",
    'download_url' : "https://github.com/PythonOptimizers/SuiteSparse.py",
    'license' : 'LGPL',
    'classifiers' : filter(None, CLASSIFIERS.split('\n')),
    'install_requires' : ['numpy', 'Cython'],
    'ext_modules' : ext_modules,
    'package_dir' : {"suitesparse": "suitesparse"},
    'packages' : packages_list,
    'zip_safe' : False

}

if use_cython:
    setup_args['cmdclass'] = {'build_ext': build_ext}

setup(**setup_args)