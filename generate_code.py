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

########################################################################################################################
# ARGUMENT PARSER
########################################################################################################################
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


def make_logger(suitesparse_config):
    # create logger
    logger_name = suitesparse_config.get('CODE_GENERATION', 'log_name')
    if logger_name == '':
        logger_name = 'suitesparse_generate_code'

    logger = logging.getLogger(logger_name)

    # levels
    log_level = LOG_LEVELS[suitesparse_config.get('CODE_GENERATION', 'log_level')]
    console_log_level = LOG_LEVELS[suitesparse_config.get('CODE_GENERATION', 'console_log_level')]
    file_log_level = LOG_LEVELS[suitesparse_config.get('CODE_GENERATION', 'file_log_level')]

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

########################################################################################################################
# CONDITIONAL CODE GENERATION
########################################################################################################################
# type of platform? 32bits or 64bits?
is_64bits = sys.maxsize > 2**32

# read suitesparse.cfg
suitesparse_config = configparser.SafeConfigParser()
suitesparse_config.read('suitesparse.cfg')

# SuiteSparse
# SPQR
SPQR_EXPERT_MODE = not suitesparse_config.getboolean('SUITESPARSE', 'NEXPERT')


# Compile for CySparse?
use_cysparse = suitesparse_config.getboolean('CODE_GENERATION', 'use_cysparse')

#####################################################
# COMMON STUFF
#####################################################

# TODO: grab this from common_types.pxd or at least from a one common file
BASIC_TYPES = ['INT32_t', 'UINT32_t', 'INT64_t', 'UINT64_t', 'FLOAT32_t', 'FLOAT64_t', 'FLOAT128_t', 'COMPLEX64_t', 'COMPLEX128_t'] #, 'COMPLEX256_t']
ELEMENT_TYPES = ['INT32_t', 'INT64_t', 'FLOAT32_t', 'FLOAT64_t']
INDEX_TYPES = ['INT32_t', 'INT64_t']
INTEGER_ELEMENT_TYPES = ['INT32_t', 'INT64_t']
REAL_ELEMENT_TYPES = ['FLOAT32_t', 'FLOAT64_t', 'FLOAT128_t']
COMPLEX_ELEMENT_TYPES = ['COMPLEX64_t', 'COMPLEX128_t'] #, 'COMPLEX256_t']

# when coding
#ELEMENT_TYPES = ['FLOAT64_t']
#ELEMENT_TYPES = ['COMPLEX64_t']
#ELEMENT_TYPES = ['COMPLEX256_t']

# Contexts
# SuiteSparse
# Umfpack
UMFPACK_INDEX_TYPES = ['INT32_t', 'INT64_t']
UMFPACK_ELEMENT_TYPES = ['FLOAT64_t', 'COMPLEX128_t']
# Cholmod
CHOLMOD_INDEX_TYPES = ['INT32_t', 'INT64_t']
CHOLMOD_ELEMENT_TYPES = ['FLOAT64_t', 'COMPLEX128_t']
# SPQR
SPQR_INDEX_TYPES = ['INT32_t', 'INT64_t']
SPQR_ELEMENT_TYPES = ['FLOAT64_t', 'COMPLEX128_t']

CYSPARSE_MATRIX_CLASSES = ['LLSparseMatrix', 'CSCSparseMatrix', 'CSRSparseMatrix']

GENERAL_CONTEXT = {
    'basic_type_list' : BASIC_TYPES,
    'type_list': ELEMENT_TYPES,
    'index_list' : INDEX_TYPES,
    'integer_list' : INTEGER_ELEMENT_TYPES,
    'real_list' : REAL_ELEMENT_TYPES,
    'complex_list' : COMPLEX_ELEMENT_TYPES,
    'umfpack_index_list': UMFPACK_INDEX_TYPES,
    'umfpack_type_list' : UMFPACK_ELEMENT_TYPES,
    'cholmod_index_list': CHOLMOD_INDEX_TYPES,
    'cholmod_type_list': CHOLMOD_ELEMENT_TYPES,
    'spqr_index_list': SPQR_INDEX_TYPES,
    'spqr_type_list': SPQR_ELEMENT_TYPES,
    'spqr_export_mode' : SPQR_EXPERT_MODE,
    'use_cysparse': use_cysparse
    }

#####################################################
# ACTION FUNCTION
#####################################################
# GENERAL
def single_generation():
    """
    Only generate one file without any suffix.
    """
    yield '', GENERAL_CONTEXT


def generate_umfpack_following_index_and_element():
    """
    Generate files following the index and element types.
    """
    for index in UMFPACK_INDEX_TYPES:
        GENERAL_CONTEXT['index'] = index
        for type in UMFPACK_ELEMENT_TYPES:
            GENERAL_CONTEXT['type'] = type
            yield '_%s_%s' % (index, type), GENERAL_CONTEXT

generate_cholmod_following_index_and_element = generate_umfpack_following_index_and_element
generate_spqr_following_index_and_element = generate_umfpack_following_index_and_element

## TESTS
def generate_umfpack_following_index_and_element_and_cysparse_matrix_classes():
    """
    Generate files following the index and element types.
    """
    for klass in CYSPARSE_MATRIX_CLASSES:
        GENERAL_CONTEXT['class'] = klass
        for index in UMFPACK_INDEX_TYPES:
            GENERAL_CONTEXT['index'] = index
            for type in UMFPACK_ELEMENT_TYPES:
                GENERAL_CONTEXT['type'] = type
                yield '_%s_%s_%s' % (index, type, klass), GENERAL_CONTEXT

########################################################################################################################
# JINJA2 FILTERS
########################################################################################################################
####################################
# UMFPACK TYPES
####################################
def cysparse_real_type_to_umfpack_family(cysparse_type):
    if cysparse_type in ['INT32_t']:
        return 'i'
    elif cysparse_type in ['INT64_t']:
        return 'l'
    elif cysparse_type in ['FLOAT64_t']:
        return 'd'
    elif cysparse_type in ['COMPLEX128_t']:
        return 'z'
    else:
        raise TypeError("Not a recognized SuiteSparse Umfpack type")

####################################
# CHOLMOD TYPES
####################################
def cysparse_real_type_to_cholmod_prefix(cysparse_type):
    if cysparse_type in ['INT32_t']:
        return 'cholmod'
    elif cysparse_type in ['INT64_t']:
        return 'cholmod_l'
    else:
        raise TypeError("Not a recognized SuiteSparse Cholmod type for prefixing Cholmod routines")

def cysparse_real_type_to_cholmod_type(cysparse_type):
    if cysparse_type in ['FLOAT32_t', 'COMPLEX64_t']:
        return 'CHOLMOD_SINGLE'
    elif cysparse_type in ['FLOAT64_t', 'COMPLEX128_t']:
        return 'CHOLMOD_DOUBLE'
    else:
        raise TypeError("Not a recognized SuiteSparse Cholmod type for prefixing Cholmod routines")


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":

    ####################################################################################################################
    # init
    ####################################################################################################################
    # line arguments
    parser = make_parser()
    arg_options = parser.parse_args()

    # create logger
    logger = make_logger(suitesparse_config=suitesparse_config)

    # cygenja engine
    current_directory = os.path.dirname(os.path.abspath(__file__))
    jinja2_env = Environment(autoescape=False,
                            loader=FileSystemLoader('/'), # we use absolute filenames
                            trim_blocks=False,
                            variable_start_string='@',
                            variable_end_string='@')

    cygenja_engine = Generator(current_directory, jinja2_env, logger=logger)

    # register filters
    # CySparse: by default
    cygenja_engine.register_common_type_filters()
    # UMFPACK
    cygenja_engine.register_filter('cysparse_real_type_to_umfpack_family', cysparse_real_type_to_umfpack_family)
    # CHOLMOD
    cygenja_engine.register_filter('cysparse_real_type_to_cholmod_prefix', cysparse_real_type_to_cholmod_prefix)
    cygenja_engine.register_filter('cysparse_real_type_to_cholmod_type', cysparse_real_type_to_cholmod_type)


    # register extensions
    cygenja_engine.register_extension('.cpy', '.py')
    cygenja_engine.register_extension('.cpx', '.pyx')
    cygenja_engine.register_extension('.cpd', '.pxd')
    cygenja_engine.register_extension('.cpi', '.pxi')

    ####################################################################################################################
    # register actions
    ####################################################################################################################
    ########## Setup ############
    cygenja_engine.register_action('config', '*.*', single_generation)
    ########## Types ############
    cygenja_engine.register_action('suitesparse/common_types', 'suitesparse_generic_types.*', single_generation)
    ########## Solvers ############
    cygenja_engine.register_action('suitesparse', 'solver.*', generate_umfpack_following_index_and_element)
    ########## UMFPACK ############
    cygenja_engine.register_action('suitesparse/umfpack', 'umfpack_solver.cpy', single_generation)
    cygenja_engine.register_action('suitesparse/umfpack', 'umfpack_solver_base.*', generate_umfpack_following_index_and_element)
    cygenja_engine.register_action('suitesparse/umfpack/cysparse_solver', '*solver.*', generate_umfpack_following_index_and_element)
    cygenja_engine.register_action('suitesparse/umfpack/generic_solver', '*solver.*', generate_umfpack_following_index_and_element)
    ########## CHOLMOD ############
    cygenja_engine.register_action('suitesparse/cholmod', 'cholmod_solver.cpy', single_generation)
    cygenja_engine.register_action('suitesparse/cholmod', 'cholmod_solver_base.*', generate_cholmod_following_index_and_element)
    cygenja_engine.register_action('suitesparse/cholmod/cysparse_solver', '*solver.*', generate_cholmod_following_index_and_element)
    cygenja_engine.register_action('suitesparse/cholmod/generic_solver', '*solver.*', generate_cholmod_following_index_and_element)
    ########## SPQR ############
    cygenja_engine.register_action('suitesparse/spqr', 'spqr_solver.cpy', single_generation)
    cygenja_engine.register_action('suitesparse/spqr', 'spqr_solver_base.*', generate_spqr_following_index_and_element)
    cygenja_engine.register_action('suitesparse/spqr/cysparse_solver', '*solver.*', generate_spqr_following_index_and_element)
    cygenja_engine.register_action('suitesparse/spqr/generic_solver', '*solver.*', generate_spqr_following_index_and_element)


    ########################################
    # TESTS
    ########################################
    ########## UMFPACK ############
    ### CySparse ###
    cygenja_engine.register_action('tests/umfpack/cysparse', '*umfpack_get_LU*', generate_umfpack_following_index_and_element_and_cysparse_matrix_classes)

    ####################################################################################################################
    # Generation
    ####################################################################################################################
    if arg_options.dry_run:
        cygenja_engine.generate(arg_options.dir_pattern, arg_options.file_pattern, action_ch='d', recursively=arg_options.recursive, force=arg_options.force)
    elif arg_options.clean:
        cygenja_engine.generate(arg_options.dir_pattern, arg_options.file_pattern, action_ch='c', recursively=arg_options.recursive, force=arg_options.force)
    else:
        cygenja_engine.generate(arg_options.dir_pattern, arg_options.file_pattern, action_ch='g', recursively=arg_options.recursive, force=arg_options.force)
        # special case for the setup.py file
        shutil.copy2(os.path.join('config', 'setup.py'), '.')