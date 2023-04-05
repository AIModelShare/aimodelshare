import os
import sys
import shutil
import tempfile
import functools
import warnings
from typing import Type


def delete_files_from_temp_dir(temp_dir_file_deletion_list):
    temp_dir = tempfile.gettempdir()
    for file_name in temp_dir_file_deletion_list:
        file_path = os.path.join(temp_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def make_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def ignore_warning(warning: Type[Warning]):
    """
    Ignore a given warning occurring during method execution.

    Args:
        warning (Warning): warning type to ignore.

    Returns:
        the inner function
    """

    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=warning)
                return func(*args, **kwargs)

        return wrapper

    return inner
