import os
import json
import logging
import sys
from time import gmtime, strftime
from typing import Any, Optional


def load_config_file(path: str) -> dict:
    """Load a JSON configuration file from the specified path and return it as a dictionary."""
    try:
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return config_dict

    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{path}': {e}")
        # Handle other potential JSON decoding errors here
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Handle other unexpected errors here
        return None


def load_json_as_list(input_file: str) -> list:
    """Load a JSON file as a list of dictionaries."""
    res = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        d = json.loads(line)
        res.append(d)
    return res


def create_directory_for_file(file_path) -> None:
    """Create the directory for the specified file path if it does not already exist."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        

def create_logger_file(log_path):
    
    if log_path is None:
        raise ValueError("Experiment path not set")
    
    # create log file
    with open(f"{log_path}", "w") as f:
        f.write("")

    log = create_logger(__name__, silent=False, to_disk=True,
                                log_file=f"{log_path}")
    
    return log

def create_logger(name, silent=False, to_disk=False, log_file=None):
    """Create a new logger"""
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("log/log_%m%d_%H%M.txt", gmtime())
        if type(log_file) == list:
            for filename in log_file:
                fh = logging.FileHandler(filename, mode='w')
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                log.addHandler(fh)
        if type(log_file) == str:
            fh = logging.FileHandler(log_file, mode='w')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)
    return log