#!/bin/python3
# coding: utf-8

""" Different functions used in several python scripts. """

# libraries
import sys
import time
from joblib.format_stack import format_exc


def log_error(path_error, source):
    """
    Function to write errors in a text file.
    :param source: list of strings
    :param path_error: string
    :return:
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    text = format_exc(exc_type, exc_value, exc_traceback, context=5,
                      tb_offset=0)
    with open(path_error, mode='a', encoding='utf-8') as f:
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


def _if_null(value, default):
    if value is None:
        return default
    return value


def _pause(duration=5):
    time.sleep(duration)


class TryMultipleTimes(object):

    def __init__(self, action=_pause, on_fail=_raise_again, n_tries=5):
        self.action_ = action
        self.n_tries_ = n_tries
        self.on_fail_ = on_fail

    def __call__(self, fun):
        def decorate(*args, **kwargs):
            n_tries = kwargs.get('n_tries', self.n_tries_)
            action = kwargs.get('action', self.action_)
            on_fail = kwargs.get('on_fail', self.on_fail_)
            error = None
            tries = 0
            while True:
                try:
                    return fun(*args, **kwargs)
                except Exception as e:
                    error = e
                    if tries == n_tries:
                        break
                tries += 1
                action(*args, **dict(kwargs, error=error))
            on_fail(error)
        return decorate
