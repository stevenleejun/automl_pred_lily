import os
import sys
import dask
import pandas as pd
import copy
from dask import compute
from dask.distributed import Client
from dask.distributed import LocalCluster
import multiprocessing as mp
from ..label_extraction.label_extraction import LabelExtraction
from .base_process import BaseProcess


class LabelExtractionProcess(BaseProcess):
    def __init__(self,
                 process_config,
                 root_config_path,
                 logger=None):
        super().__init__(
            process_config=process_config,
            root_config_path=root_config_path,
            logger=logger
        )
        self.process_config = process_config
        self.logger = logger

        self._resolve_process_config()

        self.model = None

    # 配置解析
    def _resolve_process_config(self):
        self.label_extraction_id = self.process_config.get('label_extraction_id', None)

    def init_main_class(self):
        model = LabelExtraction(
            customer_entity=self.customer_entity,
            behavior_entity=self.behavior_entity,
            product_entity=self.product_entity,
            labeling_function=self.labeling_function,
            problem_type=self.problem_type,
            actions_list=self.actions_list,
            target_para=self.target_para,
            label_type=self.label_type,

            cutoff_day=self.cutoff_day,
            cutoff_num=self.cutoff_num,
            cutoff_last_date=self.cutoff_last_date,
            lead_time_window_unit=self.lead_time_window_unit,
            lead_time_window=self.lead_time_window,
            prediction_window_unit=self.prediction_window_unit,
            prediction_window=self.prediction_window,
            sample_filtering_window_unit=self.sample_filtering_window_unit,
            sample_filtering_window=self.sample_filtering_window,
            product_label_column_name_sep=self.product_label_column_name_sep,
            customer_entity_index=self.customer_entity_index,
            behavior_entity_time_index=self.behavior_entity_time_index
        )
        return model

    def process_base(self, model=None, cutoff_last_date=None):
        if model is None:
            self.model = self.init_main_class()
            is_train = True
        else:
            self.model = model
            self.cutoff_last_date = cutoff_last_date
            is_train = False

        is_predict = not is_train

        if self.entity_set_data_url: # 用之前存储的数据
            entity_set_data_dfs_dic = self.load_entity_set_data(entity_set_data_url=self.entity_set_data_url)
        else:
            entity_set_data_dfs_dic = self.load_entity_set_data()

        entity_set_data_dfs_dic_orig = copy.deepcopy(entity_set_data_dfs_dic)

        if self.is_persist_entity_set_data:
            self.persist_entity_data(entity_set_data_dfs_dic)
        if is_train:
            self.persist_entity_data(entity_set_data_dfs_dic)

        if self.is_dask_mode:
            with LocalCluster(processes=True,
                              threads_per_worker=1,
                              # memory_limit='2GB',
                              # ip='tcp://localhost:9895',
                              ) as cluster, Client(cluster) as client:
                if self.num_workers:
                    client.cluster.scale(self.num_workers)  # ask for ten 4-thread workers
                else:
                    client.cluster.scale(int(0.8 * mp.cpu_count()))  # ask for ten 4-thread workers

            # 获取 instance_ids
            if self.behavior_entity != self.customer_entity:
                instance_ids = list(set(entity_set_data_dfs_dic[self.model.behavior_entity][self.model.customer_entity_index])
                                    & set(entity_set_data_dfs_dic[self.model.customer_entity][self.model.customer_entity_index]))
            else:
                instance_ids = list(set(entity_set_data_dfs_dic[self.model.customer_entity][self.model.customer_entity_index]))

            # 数据分区
            partitioned_dfs_dic_list = self.data_partitioned(
                entity_set_data_dfs_dic,
                instance_ids,
                customer_entity_index=self.model.customer_entity_index
            )

            base_label_result_df_list_d = []
            for partitioned_dfs_dic in partitioned_dfs_dic_list:
                entity_set_cutoff_time_dic = dask.delayed(self.get_entity_set_cutoff_time_dic)(partitioned_dfs_dic)
                tmp_base_label_result_df = dask.delayed(self.model.transform)(
                    entity_set=entity_set_cutoff_time_dic['entity_set'],
                    verbose=True,
                    logger=self.logger,
                    is_predict=is_predict,
                    cutoff_last_date=self.cutoff_last_date
                )
                base_label_result_df_list_d.append(tmp_base_label_result_df)
            base_label_result_df_list_d_new = compute(*base_label_result_df_list_d, scheduler="processes")

            self.logger.info('base_label_result_df = pd.concat(base_label_result_df_list) begin')

            if len(base_label_result_df_list_d_new) > 0: # No objects to concatenate
                base_label_result_df = pd.concat(base_label_result_df_list_d_new)
            else:
                base_label_result_df = pd.DataFrame()

        else:
            entity_set_cutoff_time_dic = self.get_entity_set_cutoff_time_dic(entity_set_data_dfs_dic)
            base_label_result_df = self.model.transform(
                entity_set=entity_set_cutoff_time_dic['entity_set'],
                verbose=True,
                logger=self.logger,
                is_predict=is_predict,
                cutoff_last_date=self.cutoff_last_date
            )
        return base_label_result_df, entity_set_data_dfs_dic_orig

    def process(self):
        base_label_result_df, entity_set_data_dfs_dic = self.process_base()

        if base_label_result_df.shape[0] == 0:
            self.logger.error('标签提取结果为空')#， 可能是由于数据集中的时间跨度不够一个prediction_window的长度！')

        self.save_model(self.model, self.label_extraction_model_file)

        build_result = self.build_result(base_label_result_df)
        save_result = self.save_result(build_result)
        result_api = self.build_api_result(save_result)
        return result_api

    def process_predictiton(self, model, cutoff_last_date=None):
        base_label_result_df, entity_set_data_dfs_dic = self.process_base(model, cutoff_last_date)

        if base_label_result_df.shape[0] == 0:
            self.logger.error('标签提取结果为空') #， 可能是由于数据集中的时间跨度不够一个prediction_window的长度！')

        result_list = self.build_result(base_label_result_df)
        return result_list, entity_set_data_dfs_dic

    def build_result(self, base_label_result_df):
        # 数据整理与保存
        result_list = []
        product_label_column_name_dict = self.model.get_product_label_column_name_dict()
        base_label_result_df = base_label_result_df.copy()
        if base_label_result_df.shape[0] == 0:
            result_list.append({
                'product_id': None,
                'label_result_df': pd.DataFrame(columns=[self.default_time_col_name, self.model.get_customer_entity_index, self.model.problem_type]),
                'label_statistics_df': pd.DataFrame(columns=['观察点(Cutoff)', '样本数', 'label=1占比', 'label=0占比']),
            })
        else:
            for product_id, product_label_column_name in product_label_column_name_dict.items():
                # 整理
                if product_id is not None:
                    if product_label_column_name not in base_label_result_df.columns:
                        self.logger.warning('product_label_column_name:{} not in base_label_result_df.columns:{}'.format(
                            product_label_column_name, base_label_result_df.columns))
                        continue

                    label_result_df = base_label_result_df[[self.model.get_customer_entity_index,
                                                            self.default_time_col_name,
                                                            product_label_column_name]]
                    label_result_df = label_result_df.rename(columns={product_label_column_name: self.model.problem_type})

                else:
                    label_result_df = base_label_result_df

                label_statistics_df = self.model.count_by_time(label_result_df)
                label_statistics_df = label_statistics_df.rename(columns=self.standard_result_colname_dic)

                label_result_df = label_result_df[[self.default_time_col_name, self.model.get_customer_entity_index, self.model.problem_type]]
                result_list.append({
                    'product_id': product_id,
                    'label_result_df': label_result_df,
                    'label_statistics_df': label_statistics_df,
                })
        return result_list

    def save_result(self, result_list):
        # 数据整理与保存
        result_save_list = []
        for result in result_list:
            # 整理
            product_id = result['product_id']
            if product_id is not None:
                project_data_dir = os.path.join(self.base_project_data_dir, product_id.replace('/', '_'))
                if not os.path.exists(project_data_dir):
                    os.makedirs(project_data_dir)
            else:
                project_data_dir = self.base_project_data_dir

            label_result_df = result['label_result_df']
            label_statistics_df = result['label_statistics_df']

            # 保存 # 结果格式化 # 标准化字段名称
            label_result_df_path = os.path.join(project_data_dir, self.label_result_df_file)
            label_result_df.to_csv(label_result_df_path, index=False)

            label_statistics_df_path = os.path.join(project_data_dir, self.label_statistics_df_file)
            label_statistics_df.to_csv(label_statistics_df_path, index=False)

            result_save_list.append({
                'product_id': product_id,
                'label_result_url': label_result_df_path,
                'label_statistics_url': label_statistics_df_path,
             })
        return result_save_list

    def build_api_result(self, result_save_list):
        # 数据整理与保存
        result = {
            'label_extraction_id': self.label_extraction_id,
            'result_list': result_save_list,
            'extract_status': 2
        }
        return result
