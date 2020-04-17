# coding=utf-8
import yaml
import os
encoding = 'utf8'


def load_yaml(yaml_file):
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r', encoding='utf-8') as infile:
            configure = yaml.load(infile, Loader=yaml.FullLoader) or {}
    else:
        configure = {}
    return configure


def get_logpath(logger):
    return logger.handlers[0].baseFilename
