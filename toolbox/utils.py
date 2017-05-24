# -*- coding: utf-8 -*-

""" Different functions used in several python scripts. """

# libraries
import sys
import os
import time
from joblib.format_stack import format_exc
from configobj import ConfigObj, ConfigObjError
from validate import Validator


def reset_log_error(path_error):
    """
    Function to reset the error directory
    :param path_error: string
    :return:
    """
    # initialize files and directories
    if not os.path.isdir(os.path.dirname(path_error)):
        os.mkdir(os.path.dirname(path_error))
    if not os.path.isdir(path_error):
        os.mkdir(path_error)
    else:
        for file in os.listdir(path_error):
            os.remove(os.path.join(path_error, file))
    return


def _get_config_spec():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'config_spec.txt')


def get_config_trace():
    """
    Function to keep a trace of the latest configuration ran
    :return:
    """
    path_input = "config.txt"
    path_output = get_config_tag("path", "general")
    with open(path_input, mode='rt', encoding='utf-8') as f:
        text = f.read()
    with open(path_output, mode='wt', encoding='utf-8') as f:
        f.write(text)
    return


def get_config_tag(tag, section):
    """
    Function to get parameters values and path from a configuration file
    :param tag: string
    :param section: string
    :return:
    """
    config = ConfigObj("config.txt",
                       encoding="utf-8",
                       configspec=_get_config_spec())
    test = config.validate(Validator())
    if test is not True:
        raise ConfigObjError("Config file validation failed.")
    return config[section][tag]


def log_error(path_error, source):
    """
    Function to write errors in a text file
    :param source: list of strings
    :param path_error: string
    :return:
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    text = format_exc(exc_type, exc_value, exc_traceback, context=5,
                      tb_offset=0)
    with open(path_error, mode='wt', encoding='utf-8') as f:
        f.write("##########################################################"
                "########### \n")
        for i in source:
            f.write(i)
            f.write("\n")
        f.write(text)
        f.write("\n")
        f.write("\n")
    return exc_type


def _do_nothing():
    pass


def _raise_again(exception):
    raise exception


def _pause(duration=5):
    time.sleep(duration)


class TryMultipleTimes(object):
    """
    Class used as decorator
    """

    def __init__(self, action=_pause, on_fail=_raise_again, n_tries=5):
        self.action_ = action
        self.n_tries_ = n_tries
        self.on_fail_ = on_fail

    def __call__(self, fun):
        def decorate(*args, **kwargs):
            n_tries = kwargs.get('n_tries', self.n_tries_)
            action = kwargs.get('action', self.action_)
            on_fail = kwargs.get('on_fail', self.on_fail_)
            tries = 0
            while True:
                try:
                    return fun(*args, **kwargs)
                except Exception as e:
                    error = e
                    if tries == n_tries:
                        break
                tries += 1
                action()
            on_fail(error)
        return decorate
