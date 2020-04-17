# encoding=utf8
import os
import sys
# sys.setdefaultencoding('utf8')
if 'a_run' not in os.getcwd().split('/')[-1]:
    os.chdir('./a_run')
# if os.getcwd().split('/')[-1] != 'a_run':
#     os.chdir('./a_run')

sys.path.append(os.path.join(os.getcwd(), '../'))

from utils.utils_common import load_yaml
from utils.utils_tools_init import initlog
from automl_pred.process_api.process_entity_set_preview import process_entity_set_preview
from automl_pred.process_api.process_feature_extraction import process_feature_extraction
from automl_pred.process_api.process_label_extraction import process_label_extraction
from automl_pred.process_api.process_model_deploying import process_model_deploying
from automl_pred.process_api.process_model_prediction import process_model_prediction
from automl_pred.process_api.process_model_training import process_model_training
from automl_pred.process_api.process_result_saving import process_result_saving

# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['Arial Unicode MS'] # 用来正常显示中文标签


def process_all(base_config_dir, project_name, sub_project_name, root_config_path):
    common_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'common.yml'))

    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'label_extraction.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'process_label_extraction')
    result = process_label_extraction(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)

    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'feature_extraction.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'process_feature_extraction')
    result = process_feature_extraction(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)

    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'model_training.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'process_model_training')
    result = process_model_training(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)


    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'model_prediction.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'process_model_prediction')
    result = process_model_prediction(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)

    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'result_saving.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'process_result_saving')
    result = process_result_saving(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)


def process_training(base_config_dir, project_name, sub_project_name, root_config_path):
    common_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'common.yml'))

    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'label_extraction.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'process_label_extraction')
    result = process_label_extraction(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)

    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'feature_extraction.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'process_feature_extraction')
    result = process_feature_extraction(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)

    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'model_training.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'process_model_training')
    result = process_model_training(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)


def process_prediction(base_config_dir, project_name, sub_project_name, root_config_path):
    common_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'common.yml'))

    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'model_prediction.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'process_model_prediction')
    result = process_model_prediction(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)

    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'result_saving.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'process_result_saving')
    result = process_result_saving(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)


def process_entity_set_preview_only(base_config_dir, project_name, sub_project_name, root_config_path):
    common_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'common.yml'))

    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'entity_set_desc.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'entity_set_desc')
    result = process_entity_set_preview(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)


def process_label_extraction_only(base_config_dir, project_name, sub_project_name, root_config_path):
    common_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'common.yml'))

    process_config = load_yaml(os.path.join(base_config_dir, project_name, sub_project_name, 'label_extraction.yml'))
    process_config.update(common_config)
    logger = initlog(project_name + '_' + sub_project_name + '_' + 'process_label_extraction')
    result = process_label_extraction(process_config, root_config_path=root_config_path, logger=logger)
    logger.info(result)
