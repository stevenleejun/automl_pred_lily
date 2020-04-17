import dask
import os
import pandas as pd
from dask import compute
from dask.distributed import Client
from dask.distributed import LocalCluster
import multiprocessing as mp

from .base_process import BaseProcess
from ..feature_extraction.feature_extraction import FeatureExtraction
from ..feature_extraction.feature_extraction import FeatureExtractionPara


class FeatureExtractionProcess(BaseProcess):
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

        self.__resolve_special_root_config()
        self._resolve_process_config()

        self.model = None

    def __resolve_special_root_config(self):
        self.default_agg_primitives = self.root_config['feature_extraction'].get('default_agg_primitives', None)
        self.default_trans_primitives = self.root_config['feature_extraction'].get('default_trans_primitives', None)
        primitives_vs_show = self.root_config['feature_extraction'].get('primitives_vs_show', {})
        self.agg_primitives_vs = primitives_vs_show.get('agg_primitives_vs', {})
        self.trans_primitives_vs = primitives_vs_show.get('trans_primitives_vs', {})
        self.default_where_primitives = self.root_config['feature_extraction'].get('default_where_primitives', None)

    def _resolve_process_config(self): #必须放在子类中，因为主类中有些时候没有这些参数
        self.feature_extraction_id = self.process_config.get('feature_extraction_id', None)
        self.label_url_list = self.process_config.get('label_url_list', self.default_label_url_list)
        if self.label_url_list is not None:
            self.label_result_url = self.label_url_list[0]['label_result_url']

        # feature_extraction_desc:
        feature_extraction_desc = self.process_config.get('feature_extraction_desc', {})
        show_agg_primitives = feature_extraction_desc.get('agg_primitives', self.default_agg_primitives)
        self.agg_primitives = [self.agg_primitives_vs.get(prim, prim) for prim in show_agg_primitives]

        show_trans_primitives = feature_extraction_desc.get('trans_primitives', self.default_trans_primitives)
        self.trans_primitives = [self.trans_primitives_vs.get(prim, prim) for prim in show_trans_primitives]

        self.ignore_entities = feature_extraction_desc.get('ignore_entities', None)
        ignore_variables = feature_extraction_desc.get('ignore_variables', None)
        self.ignore_variables = {}
        if ignore_variables:
            for ignore_variable_dic in ignore_variables:
                self.ignore_variables[ignore_variable_dic['entity_name']] = ignore_variable_dic.get('values', [])

        # 其他选项
        self.drop_contains = feature_extraction_desc.get('drop_contains', None)
        self.drop_exact = feature_extraction_desc.get('drop_exact', None)
        self.max_features = feature_extraction_desc.get('max_features', None)

        #分条件特征合成函数
        interesting_values_desc = feature_extraction_desc.get('interesting_values_desc', {})
        show_where_primitives = interesting_values_desc.get('where_primitives', self.default_where_primitives)
        self.where_primitives = [self.agg_primitives_vs.get(prim, prim) for prim in show_where_primitives]

        self.auto_max_values = interesting_values_desc.get('auto_max_values', None)
        manual_interesting_values_desc = interesting_values_desc.get('manual_interesting_values_desc', [])
        # manual_interesting_values_desc
        self.manual_interesting_values_info = dict((manual_info['entity_name'], manual_info) for manual_info in manual_interesting_values_desc)
        # # 加入product_purchase
        if self.product_entity and (self.product_list is not None) and (self.product_column is not None):
            self.manual_interesting_values_info[self.product_entity] = {
                "entity_name": self.product_entity,
                    "columns_desc": [{
                        "column_name":  self.product_column,
                        "column_values": self.product_list
                    }]
                }

    def init_main_class(self):
        # 构建 FeatureExtractionPara
        model = FeatureExtractionPara(
            customer_entity=self.customer_entity,
            customer_entity_index=self.customer_entity_index,
            agg_primitives=self.agg_primitives,
            trans_primitives=self.trans_primitives,
            ignore_entities=self.ignore_entities,
            ignore_variables=self.ignore_variables,
            training_window_unit=self.training_window_unit,
            training_window=self.training_window,
            entity_set_drop_index_list=self.entity_set_drop_index_list,
            auto_max_values=self.auto_max_values,
            manual_interesting_values_info=self.manual_interesting_values_info,
            where_primitives=self.where_primitives,
            drop_contains=self.drop_contains,
            drop_exact=self.drop_exact,
        )
        return model

    def process_base(self, model=None, **kwargs):
        # 构建 model
        if model is None:
            self.model = self.init_main_class()
            is_train = True
        else:
            self.model = model
            is_train = False

        feature_extraction_model = FeatureExtraction(
            self.model
        )

        # 数据读取
        if 'entity_set_data_dfs_dic' in kwargs.keys():
            entity_set_data_dfs_dic = kwargs['entity_set_data_dfs_dic']
        else:
            entity_set_data_dfs_dic = self.load_entity_set_data(is_load_before=True)

        if 'cutoff_time' in kwargs.keys():
            cutoff_time = kwargs['cutoff_time']
        else:
            try:
                cutoff_time = pd.read_csv(self.label_result_url)
            except:
                cutoff_time = pd.DataFrame(columns=[self.default_time_col_name, self.customer_entity_index, self.problem_type])

        entity_set_data_dfs_dic['cutoff_time'] = cutoff_time

        # is_dask_mode
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
                instance_ids = cutoff_time[self.model.customer_entity_index].unique()

                # 数据分区
                partitioned_dfs_dic_list = self.data_partitioned(
                    entity_set_data_dfs_dic,
                    instance_ids,
                    customer_entity_index=self.model.customer_entity_index
                )

                feature_result_df_list = []
                for partitioned_dfs_dic in partitioned_dfs_dic_list:
                    entity_set_cutoff_time_dic = dask.delayed(self.get_entity_set_cutoff_time_dic)(
                        data_dfs_dic=partitioned_dfs_dic,
                    )
                    tmp_feature_result_df = dask.delayed(feature_extraction_model.transform)(
                        entity_set=entity_set_cutoff_time_dic['entity_set'],
                        cutoff_time=entity_set_cutoff_time_dic['cutoff_time'],
                        logger=self.logger,
                        is_train=is_train
                    )
                    feature_result_df_list.append(tmp_feature_result_df)
                # feature_result_df_list = dask.delayed(feature_result_df_list)
                # feature_result_df_list = feature_result_df_list.compute(scheduler='threads', num_workers=self.num_workers)
                feature_result_df_list_new = compute(*feature_result_df_list, scheduler="processes")

                self.logger.info('feature_result_df = pd.concat(feature_result_df_list) begin')
                if len(feature_result_df_list_new) > 0:
                    feature_result_df = pd.concat(feature_result_df_list_new)
                else:
                    feature_result_df = pd.DataFrame()

        else:
            entity_set_cutoff_time_dic = self.get_entity_set_cutoff_time_dic(
                data_dfs_dic=entity_set_data_dfs_dic,
            )
            feature_result_df = feature_extraction_model.transform(
                entity_set=entity_set_cutoff_time_dic['entity_set'],
                cutoff_time=entity_set_cutoff_time_dic['cutoff_time'],
                logger=self.logger,
                is_train=is_train
            )

        if feature_result_df.shape[0] > 0:
            feature_result_df = self.model.process_feature_engineering(
                is_train=is_train,
                feature_result_df=feature_result_df,
                project_id=self.project_id
            )
        # # 刷新基础模型中的参数
        # # bug  Need to specify at least one of 'labels', 'index' or 'columns' dask中间不会更新数据
        # self.model = feature_extraction_model.refresh_base_model()
        return feature_result_df, self.model

    def process(self):
        feature_result_df, self.model = self.process_base()
        self.save_model(self.model, self.feature_extraction_para_model_file)
        result = self.build_result(feature_result_df)
        result = self.save_result(result)
        result_api = self.build_api_result(result)
        return result_api
    
    def process_predictiton(self, model, **kwargs):
        feature_result_df, _ = self.process_base(model, **kwargs)
        result = self.build_result(feature_result_df)
        feature_result_df = result['feature_result_df']
        return feature_result_df
    
    def build_result(self, feature_result_df):
        # 结果保存
        feature_statistics_df = self.model.stat_result(feature_result_df)
        result = {
            "feature_result_df": feature_result_df,
            "feature_statistics_df": feature_statistics_df,
        }
        return result

    def save_result(self, result):
        # 结果保存
        feature_result_df = result['feature_result_df']
        feature_statistics_df = result['feature_statistics_df']

        feature_result_df_path = self.default_feature_result_url
        self.logger.info('feature_result_df.to_csv(feature_result_df_path, index=False) begin')
        # feature_result_df.to_csv(feature_result_df_path, index=False)
        # feature_result_df.to_pickle(feature_result_df_path)
        feature_result_df.to_hdf(feature_result_df_path, key='default')

        feature_statistics_df_path = self.default_feature_statistics_url
        self.logger.info('feature_statistics_df.to_csv(feature_statistics_df_path, index=False) begin')
        feature_statistics_df.to_csv(feature_statistics_df_path, encoding="utf-8", index=False)

        result = {
            "feature_result_url": feature_result_df_path,
            "feature_statistics_url": feature_statistics_df_path,
        }
        return result

    def build_api_result(self, result):
        # 结果保存
        result.update({
            "feature_extraction_id": self.feature_extraction_id,
            "extract_status": 2
        })
        return result
