# encoding=utf8
import os
import sys
import argparse
# sys.setdefaultencoding('utf8')
if 'a_run' not in os.getcwd().split('/')[-1]:
    os.chdir('./a_run')

sys.path.append(os.path.join(os.getcwd(), '../'))

from utils.utils_common import load_yaml
from utils.utils_tools_init import initlog
from comment_analysis.comment_analysis import comment_anlysis


if __name__ == '__main__':
    # 参数读取
    ap = argparse.ArgumentParser()
    ap.add_argument('-begin_date', '--begin_date',  default='0000-00-00', help='补跑开始日期')
    ap.add_argument('-end_date', '--end_date',  default='0000-00-00', help='补跑结束日期')
    args = (ap.parse_args())
    begin_date = args.begin_date
    end_date = args.end_date

    # 配置解析
    root_config = load_yaml('../config/comment_analysis/root_config.yml')

    #logger
    logger = initlog('comment_analysis')

    comment_anlysis(logger, root_config, begin_date, end_date)