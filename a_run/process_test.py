# encoding=utf8
import os
import sys
# sys.setdefaultencoding('utf8')
if 'a_run' not in os.getcwd().split('/')[-1]:
    os.chdir('./a_run')

sys.path.append(os.path.join(os.getcwd(), '../'))
from tools_ml import process_label_extraction_only
from tools_ml import process_prediction
from tools_ml import process_training
from tools_ml import process_all


if __name__ == '__main__':

    base_config = '../config/config_lily_ml/'
    root_config_path = os.path.join(base_config, 'root_config.yml')
    project_name = 'lily'
    sub_project_name = 'product_purchase'

    process_training(base_config, project_name, sub_project_name, root_config_path)

