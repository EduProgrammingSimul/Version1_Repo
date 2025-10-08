import logging


logger = logging.getLogger(__name__)


DEPRECATION_MESSAGE = """
*******************************************************************************
* *
* WARNING: The 'flc_tuner.py' module is DEPRECATED.                         *
* *
* All FLC optimization should be performed using the robust, multi-scenario *
* optimizer located in:                                                     *
* *
* optimization_suite/flc_optimizer.py                                   *
* *
* Please update any scripts or workflows to call the functions within that  *
* module instead. This file will be removed in a future version.            *
* *
*******************************************************************************
"""


def _raise_deprecation_warning():

    logger.error(DEPRECATION_MESSAGE)


_raise_deprecation_warning()


def tune_flc_scaling(*args, **kwargs):

    _raise_deprecation_warning()
    print("DEPRECATED: Please use optimize_flc_scaling_de from flc_optimizer.py")
    return None


def save_flc_params(*args, **kwargs):

    _raise_deprecation_warning()
    print("DEPRECATED: Please use the save functionality within flc_optimizer.py")
    return False
