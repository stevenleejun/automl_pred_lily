# coding: utf-8
import os
import logging
import datetime
from .utils_common import load_yaml

def create_dirs(config):
    for ite in config:
        if '_dir' == ite:
            for dir in config[ite]:
                if not os.path.exists(dir):
                    os.mkdir(dir)

config_file = os.path.join('../config', 'config_init.yml')
config_init = load_yaml(config_file)
create_dirs(config_init)
level = config_init['logger']['level']
logger_dir = config_init['logger']['logger_dir']

def initlog(jobname='', log_dir=logger_dir, log_path=None, level=level):
    """
    :param jobname: 工程名
    :return: 日志对象
    """
    log_dir=os.path.join(log_dir, datetime.datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logger = logging.getLogger(jobname)
    if log_path is None:
        log_path = os.path.join(log_dir, '%s_%s.log' % (datetime.datetime.now().strftime('%X').replace(':', '_'), jobname))
    hdlr = logging.FileHandler(log_path, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s  %(message)s')
    # formatter = logging.Formatter('%(asctime)s  %(filename)s : %(levelname)s  %(message)s')
    hdlr.setFormatter(formatter)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(hdlr)
    logger.setLevel(level=level)
    return logger
