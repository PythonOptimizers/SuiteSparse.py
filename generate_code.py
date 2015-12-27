#!/usr/bin/env python
###############################################################################
# This script generates all templated code for SuiteSparse.py
# It this the single one script to use before Cythonizing the SuiteSparse.py library.
# This script is NOT automatically called by setup.py
#
# We use our internal library cygenja, using itself the Jinja2 template engine:
# http://jinja.pocoo.org/docs/dev/
#
###############################################################################

from cygenja.generator import Generator
from jinja2 import Environment, FileSystemLoader

import configparser
import argparse
import os
import sys
import shutil
import logging

#####################################################
# PARSER
#####################################################
def make_parser():
    """
    Create a comment line argument parser.

    Returns:
        The command line parser.
    """
    parser = argparse.ArgumentParser(description='%s: a Cython code generator for the SuiteSparse.py library' % os.path.basename(sys.argv[0]))
    parser.add_argument("-r", "--recursive", help="Act recursively", action='store_true', required=False)
    parser.add_argument("-c", "--clean", help="Clean generated files", action='store_true', required=False)
    parser.add_argument("-d", "--dry_run", help="Dry run: no action is taken", action='store_true', required=False)
    parser.add_argument("-f", "--force", help="Force generation no matter what", action='store_true', required=False)

    parser.add_argument('dir_pattern', nargs='?', default='.', help='Glob pattern')
    parser.add_argument('file_pattern', nargs='?', default='*.*', help='Fnmatch pattern')

    return parser

###################################################################s####################################################
# LOGGING
########################################################################################################################
LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }


def make_logger(cysparse_config):
    # create logger
    logger_name = cysparse_config.get('CODE_GENERATION', 'log_name')
    if logger_name == '':
        logger_name = 'suitesparse_generate_code'

    logger = logging.getLogger(logger_name)

    # levels
    log_level = LOG_LEVELS[cysparse_config.get('CODE_GENERATION', 'log_level')]
    console_log_level = LOG_LEVELS[cysparse_config.get('CODE_GENERATION', 'console_log_level')]
    file_log_level = LOG_LEVELS[cysparse_config.get('CODE_GENERATION', 'file_log_level')]

    logger.setLevel(log_level)

    # create console handler and set logging level
    ch = logging.StreamHandler()
    ch.setLevel(console_log_level)

    # create file handler and set logging level
    log_file_name = logger_name + '.log'
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(file_log_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

#######################################
# CONDITIONAL CODE GENERATION
#######################################
# type of platform? 32bits or 64bits?
is_64bits = sys.maxsize > 2**32

# read cysparse.cfg
cysparse_config = configparser.SafeConfigParser()
cysparse_config.read('suitesparse.cfg')


# index type for LLSparseMatrix
DEFAULT_INDEX_TYPE = 'INT32_T'
DEFAULT_ELEMENT_TYPE = 'FLOAT32_T'
if is_64bits:
    DEFAULT_INDEX_TYPE = 'INT64_T'
    DEFAULT_ELEMENT_TYPE = 'FLOAT64_T'


if cysparse_config.get('CODE_GENERATION', 'DEFAULT_INDEX_TYPE') == '32bits':
    DEFAULT_INDEX_TYPE = 'INT32_T'
elif cysparse_config.get('CODE_GENERATION', 'DEFAULT_INDEX_TYPE') == '64bits':
    DEFAULT_INDEX_TYPE = 'INT64_T'
else:
    # don't do anything: use platform's default
    pass

if cysparse_config.get('CODE_GENERATION', 'DEFAULT_ELEMENT_TYPE') == '32bits':
    DEFAULT_ELEMENT_TYPE = 'FLOAT32_T'
elif cysparse_config.get('CODE_GENERATION', 'DEFAULT_ELEMENT_TYPE') == '64bits':
    DEFAULT_ELEMENT_TYPE = 'FLOAT64_T'
else:
    # don't do anything: use platform's default
    pass

#####################################################
# COMMON STUFF
#####################################################

# ======================================================================================================================
# As of 22nd of December 2015, we no longer support COMPLEX256_T as it is too problematic to work with in Cython.
#
# ======================================================================================================================

# TODO: grab this from common_types.pxd or at least from a one common file
BASIC_TYPES = ['INT32_t', 'UINT32_t', 'INT64_t', 'UINT64_t', 'FLOAT32_t', 'FLOAT64_t', 'FLOAT128_t', 'COMPLEX64_t', 'COMPLEX128_t'] #, 'COMPLEX256_t']
ELEMENT_TYPES = ['INT32_t', 'INT64_t', 'FLOAT32_t', 'FLOAT64_t', 'FLOAT128_t', 'COMPLEX64_t', 'COMPLEX128_t'] #, 'COMPLEX256_t']
INDEX_TYPES = ['INT32_t', 'INT64_t']
INTEGER_ELEMENT_TYPES = ['INT32_t', 'INT64_t']
REAL_ELEMENT_TYPES = ['FLOAT32_t', 'FLOAT64_t', 'FLOAT128_t']
COMPLEX_ELEMENT_TYPES = ['COMPLEX64_t', 'COMPLEX128_t'] #, 'COMPLEX256_t']

# Matrix market types
MM_INDEX_TYPES = ['INT32_t', 'INT64_t']
MM_ELEMENT_TYPES = ['INT64_t', 'FLOAT64_t'] #, 'COMPLEX128_t']

# when coding
#ELEMENT_TYPES = ['FLOAT64_t']
#ELEMENT_TYPES = ['COMPLEX64_t']
#ELEMENT_TYPES = ['COMPLEX256_t']

GENERAL_CONTEXT = {
                    'basic_type_list' : BASIC_TYPES,
                    'type_list': ELEMENT_TYPES,
                    'index_list' : INDEX_TYPES,
                    'default_index_type' : DEFAULT_INDEX_TYPE,
                    'default_element_type' : DEFAULT_ELEMENT_TYPE,
                    'integer_list' : INTEGER_ELEMENT_TYPES,
                    'real_list' : REAL_ELEMENT_TYPES,
                    'complex_list' : COMPLEX_ELEMENT_TYPES,
                    'mm_index_list' : MM_INDEX_TYPES,
                    'mm_type_list' : MM_ELEMENT_TYPES,
                  }
