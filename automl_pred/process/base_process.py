import os
import pickle
import sys
import re
import shutil
import logging
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from utils.utils_common import load_yaml
from ..label_extraction.label_fun import churn
from ..label_extraction.label_fun import purchase
from ..label_extraction.label_fun import product_purchase
from ..label_extraction.label_fun import value
from ..label_extraction.label_fun import highvalue

from ..entity_set_build.entity_set_build import EntitySetBuild
from ..load_data.entity_set_data_load import EntitySetDataLoad
import datetime


class BaseProcess:
    _support_problem_type = ['churn', 'purchase', 'value', 'product_purchase']

    def __init__(self,
                 process_config,
                 root_config_path,
                 logger=None):
        self.root_config_path = root_config_path
        self.process_config = process_config
        self.logger = logger
        self.root_config_path = root_config_path

        # 为了后续深一层解析
        self.behavior_entity = None
        self.customer_entity = None
        self.product_entity = None
        self.entity_set_info = None
        self.problem_type = None
        self.project_id = None
        self.cutoff_last_date = None
        self.product_list = []

        self._resolve_root_config()
        self._resolve_project_id_config()
        self._resolve_business_desc_config()
        self._resolve_entity_set_desc_config()
        self._resolve_label_desc_config()
        self._resolve_entity_set_data_config()

        self._get_base_dir()
        self._get_default_file_path()
        self._get_target_para()
        self._get_entity_index_para()

        self._create_project_path()

        self._rely_on_logger()

    def _rely_on_logger(self):
        logger_process_config = self.process_config
        pattern = re.compile(r'[\'\"]password[\'\"].*?,')
        logger_process_config = pattern.sub(r"'password': '******',", str(logger_process_config))
        self.logger.info('配置信息为:{}'.format(logger_process_config))
        self.logger_file = open(self.logger.handlers[0].baseFilename, 'a')
        sys.stdout = self.logger_file
        # sys.stderr = self.logger_file

    # 基础配置
    def _resolve_root_config(self):
        root_config = load_yaml(self.root_config_path)
        self.root_config = root_config
        # root
        self.product_label_column_name_sep = root_config['product_label_column_name_sep']
        self.standard_result_colname_dic = root_config['standard_result_colname_dic']
        self.problem_type_vs_label_type = root_config['problem_type_vs_label_type']
        self.ana_type_vs_data_type = root_config['ana_type_vs_data_type']
        self.default_time_col_name = root_config.get('default_time_col_name', 'time')

        # dir定义
        self.base_project_dir = root_config['dir']['base_project_dir']
        self.sub_project_models_dir = root_config['dir']['sub_project_models_dir']
        self.sub_project_data_dir = root_config['dir']['sub_project_data_dir']
        self.sub_project_srcdata_dir = root_config['dir']['sub_project_srcdata_dir']
        self.sub_project_models_prediction_data_dir = root_config['dir']['sub_project_models_prediction_data_dir']
        self.base_project_dir_deploying = root_config['dir']['base_project_dir_deploying']

        # file_map
        self.label_extraction_model_file = root_config['file_map'].get('label_extraction_model', 'label_extraction_model.pkl')
        self.label_result_df_file = root_config['file_map'].get('label_result_df', 'label_result_df.csv')
        self.label_statistics_df_file = root_config['file_map'].get('label_statistics_df', 'label_statistics_df.csv')
        self.feature_result_df_file = root_config['file_map'].get('feature_result_df', 'feature_result_df.faturedf')
        self.feature_statistics_df_file = root_config['file_map'].get('feature_statistics_df', 'feature_statistics_df.csv')
        self.feature_extraction_para_model_file = root_config['file_map'].get(
            'feature_extraction_para_model',
            'feature_extraction_para_model.pkl'
        )
        self.model_training_model_file = root_config['file_map'].get('model_training_model', 'model_training_model.pkl')
        self.prediction_result_df_file = root_config['file_map'].get('prediction_result_df', 'prediction_result_df.csv')
        self.entity_set_data_dfs_dic_file = root_config['file_map'].get(
            'entity_set_data_dfs_dic',
            'entity_set_data_dfs_dic.pkl'
        )

        # model_training
        self.model_eval_key_value = root_config['model_training'].get('model_eval_key_value', {})
        self.model_split_key_value = root_config['model_training'].get('model_split_key_value', {})

        # run_para
        self.is_dask_mode = root_config['run_para']['is_dask_mode']
        self.test_mode = root_config['run_para'].get('test_mode', None)
        self.test_nrows = root_config['run_para'].get('test_nrows', None)

        # dask
        self.partitioned_chunksize = root_config['dask']['partitioned_chunksize']
        self.num_workers = root_config['dask'].get('num_workers', None)

        if self.test_mode:
            self.nrows = self.test_nrows
        else:
            self.nrows = None

    def _resolve_entity_set_data_config(self):
        entity_set_data_desc = self.process_config.get('entity_set_data_desc', {})
        entity_set_data_url_project_id = entity_set_data_desc.get('entity_set_data_url_project_id', None)
        if entity_set_data_url_project_id:
            # ../../pred/test_churn/srcdata/entity_set_data_dfs_dic.pkl
            self.default_entity_set_data_url = os.path.join(
                self.base_project_dir,
                entity_set_data_url_project_id,
                self.sub_project_srcdata_dir,
                self.entity_set_data_dfs_dic_file
            )
        else:
            self.default_entity_set_data_url = None
        self.entity_set_data_url = entity_set_data_desc.get('entity_set_data_url', self.default_entity_set_data_url)
        self.is_persist_entity_set_data = entity_set_data_desc.get('is_persist_entity_set_data', False)

    def _resolve_business_desc_config(self):
        # business_desc
        business_desc = self.process_config.get('business_desc', {})
        if len(business_desc) > 0:
            self.problem_type = business_desc.get('problem_type', None)
            self.customer_entity = business_desc.get('customer_entity', None)
            self.behavior_entity = business_desc.get('behavior_entity', None)
            self.product_entity = business_desc.get('product_entity', None)
            self.label_type = self.problem_type_vs_label_type.get(self.problem_type, None)

            target_definition = business_desc.get('target_definition', {})
            self.actions_list = target_definition.get('actions_list', [])
            self.churn_para = target_definition.get('churn_para', None)
            self.product_purchase_para = target_definition.get('product_purchase_para', {})
            self.product_list = self.product_purchase_para.get('product_list', [])
            self.product_column = self.product_purchase_para.get('product_column', None)

            self.value_para = target_definition.get('value_para', None)
            self.highvalue_para = target_definition.get('highvalue_para', None)

    def _resolve_entity_set_desc_config(self):
        entity_set_desc = self.process_config.get('entity_set_desc', None)
        # 转化成字典格式
        if entity_set_desc is None:
            return

        entity_set_info = {}
        for entity_desc in entity_set_desc:
            entity_set_info[entity_desc['entity_name']] = entity_desc
        self.entity_set_info = entity_set_info

        entity_set_drop_index_list = []
        for entity_name, entity_name_info in entity_set_info.items():
            index = entity_name_info.get('index', [])
            entity_set_drop_index_list.extend(index)
        self.entity_set_drop_index_list = list(set(entity_set_drop_index_list))

    def _resolve_label_desc_config(self):
        label_desc = self.process_config.get('label_desc', {})
        if len(label_desc) > 0:
            cutoff_point = label_desc.get('cutoff_point', {})
            self.cutoff_day = cutoff_point.get('cutoff_day', None)
            self.cutoff_num = cutoff_point.get('cutoff_num', None)
            self.cutoff_last_date = cutoff_point.get('cutoff_last_date', None)

            sample_filtering_window = label_desc.get('sample_filtering_window', {})
            self.sample_filtering_window_unit = sample_filtering_window.get('sample_filtering_window_unit', None)
            self.sample_filtering_window = sample_filtering_window.get('sample_filtering_window', None)

            lead_time_window = label_desc.get('lead_time_window', {})
            self.lead_time_window_unit = lead_time_window.get('lead_time_window_unit', None)
            self.lead_time_window = lead_time_window.get('lead_time_window', None)

            prediction_window = label_desc.get('prediction_window', {})
            self.prediction_window_unit = prediction_window.get('prediction_window_unit', None)
            self.prediction_window = prediction_window.get('prediction_window', None)

            training_window = label_desc.get('training_window', {})
            self.training_window_unit = training_window.get('training_window_unit', None)
            self.training_window = training_window.get('training_window', None)

    def _resolve_project_id_config(self):
        # label_extraction_id
        self.orig_project_id = self.process_config.get('project_id', None)
        self.project_id = self.orig_project_id or 'project_id_' + str(datetime.datetime.now()).replace(' ', '_')

    def _get_base_dir(self):
        project_dir = os.path.join(self.base_project_dir, self.project_id)
        self.model_training_id = self.process_config.get('model_training_id', 'no_model_training_id')
        self.base_project_models_dir = os.path.join(project_dir, self.sub_project_models_dir, self.model_training_id)
        self.base_project_data_dir = os.path.join(project_dir, self.sub_project_data_dir)
        self.base_project_srcdata_dir = os.path.join(project_dir, self.sub_project_srcdata_dir)
        self.project_dir = project_dir

    def _copy_before_model_file(self):
        if self.model_training_id != 'no_model_training_id':
            before_base_project_models_dir = os.path.join(self.project_dir, self.sub_project_models_dir, 'no_model_training_id')
            if os.path.exists(before_base_project_models_dir):
                if os.path.exists(self.base_project_models_dir):
                    shutil.rmtree(self.base_project_models_dir)
                shutil.copytree(before_base_project_models_dir, self.base_project_models_dir)

    def _get_default_file_path(self):
        # label_url_list:
        #   - product_id:
        #     label_result_url: ../../pred/test_churn/data/label_result_df.csv
        self.default_label_url_list = []
        for product_id in self.product_list:
            tmp_label_url = {
                'product_id': product_id,
                'label_result_url': os.path.join(
                    self.base_project_data_dir,
                    str(product_id).replace('/', '_'),
                    self.label_result_df_file
                )
            }
            self.default_label_url_list.append(tmp_label_url)
        if len(self.product_list) == 0:
            tmp_label_url = {
                'product_id': None,
                'label_result_url': os.path.join(
                    self.base_project_data_dir,
                    self.label_result_df_file
                )
            }
            self.default_label_url_list.append(tmp_label_url)

        # feature_result_url:../../ pred / test_churn / data / feature_result_df.csv
        self.default_feature_result_url = os.path.join(
            self.base_project_data_dir,
            self.feature_result_df_file
        )

        self.default_feature_statistics_url = os.path.join(
            self.base_project_data_dir,
            self.feature_statistics_df_file
        )
        # model_url: '../../pred_deploying/test_churn/models'
        self.default_model_url = self.base_project_models_dir

        # ../../pred_deploying/test_purchase/models/prediction/test_purchase__prediction_result_df.csv
        self.default_prediction_result_url = os.path.join(self.default_model_url, self.project_id + '__' + self.prediction_result_df_file)

    def _get_target_para(self):
        # 获取不同业务问题的相关参数
        if self.problem_type is not None:
            if self.problem_type == 'churn':
                self.target_para = self.churn_para
                self.labeling_function = churn
            elif self.problem_type == 'product_purchase':
                self.target_para = self.product_purchase_para
                self.labeling_function = product_purchase
            elif self.problem_type == 'highvalue':
                self.target_para = self.highvalue_para
                self.labeling_function = highvalue
            elif self.problem_type == 'purchase':
                self.target_para = {}
                self.labeling_function = purchase
            elif self.problem_type == 'value':
                self.target_para = self.value_para
                self.labeling_function = value
            else:
                self.labeling_function = None
                error_info = 'self.problem_type:{} 没有定义，现在只支持:{}中的一个'.format(
                    self.problem_type,
                    self._support_problem_type)
                self.logger.error(error_info)

    def _get_entity_index_para(self):
        if (self.entity_set_info is not None) and (self.customer_entity is not None):
            self.customer_entity_index = self.entity_set_info[self.customer_entity]['index'][0]

        if (self.entity_set_info is not None) and (self.behavior_entity is not None):
            self.behavior_entity_time_index = self.entity_set_info[self.behavior_entity]['time_index']

    def _create_project_path(self):
        # # 创建project_path
        if self.orig_project_id is not None:
            if not os.path.exists(self.base_project_models_dir):
                os.makedirs(self.base_project_models_dir)
            if not os.path.exists(self.base_project_data_dir):
                os.makedirs(self.base_project_data_dir)
            if not os.path.exists(self.base_project_srcdata_dir):
                os.makedirs(self.base_project_srcdata_dir)

    def get_entity_set_cutoff_time_dic(self, data_dfs_dic, auto_max_values=None, manual_interesting_values_info=None):
        # # 正式逻辑
        cutoff_time = None
        if 'cutoff_time' in data_dfs_dic.keys():
            cutoff_time = data_dfs_dic.pop('cutoff_time')

        entity_set_build = EntitySetBuild(
            entity_set_info=self.entity_set_info,
            data_dfs_dic=data_dfs_dic,
            logger=self.logger,
            auto_max_values=auto_max_values,
            manual_interesting_values_info=manual_interesting_values_info
        )
        entity_set = entity_set_build.transform()
        entity_set_cutoff_time_dic = {
            'entity_set': entity_set,
            'cutoff_time': cutoff_time
        }
        return entity_set_cutoff_time_dic

    def load_entity_set_data(self, is_load_before=False, entity_set_data_url=None):
        if is_load_before:
            entity_set_data_dfs_dic_path = os.path.join(self.base_project_srcdata_dir,
                                                        self.entity_set_data_dfs_dic_file)
            with open(entity_set_data_dfs_dic_path, 'rb') as file:
                entity_set_data_dfs_dic = pickle.load(file)

        elif entity_set_data_url:
            with open(entity_set_data_url, 'rb') as file:
                entity_set_data_dfs_dic = pickle.load(file)

        else:
            load_entity_set_data_model = EntitySetDataLoad(
                entity_set_info=self.entity_set_info,
                nrows=self.nrows,
                ana_type_vs_data_type=self.ana_type_vs_data_type,
                logger=self.logger
            )
            entity_set_data_dfs_dic = load_entity_set_data_model.process()

        return entity_set_data_dfs_dic

    def data_partitioned(
            self,
            data_dfs_dic,
            instance_ids,
            customer_entity_index,
            **kwargs
    ):
        partitioned_dfs_dic_list = []
        for i in tqdm(range(0, len(instance_ids), self.partitioned_chunksize)):
            ids_keep = instance_ids[i: i + self.partitioned_chunksize]
            partitioned_dfs_dic = {}
            for table_df_name, table_df in data_dfs_dic.items():
                if customer_entity_index in table_df.columns:
                    table_sample = table_df[table_df[customer_entity_index].isin(ids_keep)]
                else:
                    table_sample = table_df
                partitioned_dfs_dic[table_df_name] = table_sample

            for table_df_name, table_df in kwargs.items():
                if customer_entity_index in table_df.columns:
                    table_sample = table_df[table_df[customer_entity_index].isin(ids_keep)]
                else:
                    table_sample = table_df
                partitioned_dfs_dic[table_df_name] = table_sample

            partitioned_dfs_dic_list.append(partitioned_dfs_dic)
        return partitioned_dfs_dic_list

    @staticmethod
    def init_dask_client():
        pbar = ProgressBar()
        pbar.register()

        cluster = LocalCluster()
        client = Client(cluster)
        return cluster, client

    def persist_entity_data(self, data_dfs_dic):
        data_dfs_dic_path = os.path.join(self.base_project_srcdata_dir, self.entity_set_data_dfs_dic_file)
        with open(data_dfs_dic_path, 'wb') as file:
            pickle.dump(data_dfs_dic, file)

    def save_model(
            self,
            model,
            model_file_name,
            base_model_dir=None,
            sub_model_dir=None
    ):
        # 模型保存
        if base_model_dir is None:
            base_model_dir = self.base_project_models_dir

        if sub_model_dir is not None:
            model_dir = os.path.join(base_model_dir, sub_model_dir)
        else:
            model_dir = base_model_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, model_file_name)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

    def load_model(
            self,
            model_file_name,
            base_model_dir=None,
            sub_model_dir=None
    ):
        if base_model_dir is None:
            base_model_dir = self.base_project_models_dir

        if sub_model_dir is not None:
            model_dir = os.path.join(base_model_dir, sub_model_dir)
        else:
            model_dir = base_model_dir

        model_path = os.path.join(model_dir, model_file_name)

        if not os.path.exists(model_path):
            self.logger.error('model_training_model_path:{} not exists'.format(model_path))
            return None

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        return model

    def save_csv(
            self,
            data_df,
            data_file_name,
            base_data_dir=None,
            sub_data_dir=None
    ):
        # 模型保存
        if base_data_dir is None:
            base_data_dir = self.base_project_data_dir

        if sub_data_dir is not None:
            data_dir = os.path.join(base_data_dir, sub_data_dir)
        else:
            data_dir = base_data_dir

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_path = os.path.join(data_dir, data_file_name)
        data_path = os.path.abspath(data_path)
        data_df.to_csv(data_path, index=False)
        return data_path

    def __resolve_special_root_config(self):
        pass

    def _resolve_process_config(self):
        pass

    def init_main_class(self):
        pass

    def process(self):
        pass

    def process_predictiton(self, model, **kwargs):
        pass

    def process_base(self, model=None, **kwargs):
        pass

    def build_result(self, process_base_result):
        return process_base_result

    def save_result(self, build_result):
        return build_result

    def build_api_result(self, save_result):
        return save_result
