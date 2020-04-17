import os
import pandas as pd
import sys

from .base_process import BaseProcess
from .label_extraction_process import LabelExtractionProcess
from .feature_extraction_process import FeatureExtractionProcess


class ModelPredictionProcess(BaseProcess):
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

    def _resolve_process_config(self):
        # 配置解析
        model_url = self.process_config.get('model_url', self.default_model_url)
        self.base_project_models_dir = model_url
        self.prediction_id = self.process_config.get('prediction_id', None)

        model_prediction_desc = self.process_config.get('model_prediction_desc', {})
        self.cutoff_last_date = model_prediction_desc.get('cutoff_last_date', None)
        # self.customer_entity = None
        # self.customer_entity_index = None

        if self.project_id is not None:
            self.prediction_result_df_file = self.project_id + '__' + self.prediction_result_df_file

    def process_base(self, **kwargs):
        # 验证 project_path
        if not os.path.exists(self.base_project_models_dir):
            self.logger.error('model_url:{} not exists'.format(self.base_project_models_dir))
            return None

        # label_extraction_model
        label_extraction_model = self.load_model(
            model_file_name=self.label_extraction_model_file,
            base_model_dir=self.base_project_models_dir)

        self.process_config['behavior_entity'] = label_extraction_model.behavior_entity
        self.process_config['customer_entity'] = label_extraction_model.customer_entity
        self.process_config['customer_entity_index'] = label_extraction_model.customer_entity_index

        label_extraction_process_model = LabelExtractionProcess(
            process_config=self.process_config,
            root_config_path=self.root_config_path,
            logger=self.logger
        )

        label_extraction_result_list,\
            entity_set_data_dfs_dic = label_extraction_process_model.process_predictiton(
                model=label_extraction_model,
                cutoff_last_date=self.cutoff_last_date
            )
        if len(label_extraction_result_list) == 0:
            self.logger.error('标签结果为空, len(label_extraction_result_list) == 0')
            prediction_result_df = pd.DataFrame()
            return prediction_result_df

        # feature_extraction_model
        feature_extraction_para_model = self.load_model(
            model_file_name=self.feature_extraction_para_model_file,
            base_model_dir=self.base_project_models_dir)

        feature_extraction_process_model = FeatureExtractionProcess(
            process_config=self.process_config,
            root_config_path=self.root_config_path,
            logger=self.logger
        )
        feature_extraction_process_kwargs = {
            'entity_set_data_dfs_dic': entity_set_data_dfs_dic,
            'cutoff_time': label_extraction_result_list[0]['label_result_df'],
        }

        feature_result_df = feature_extraction_process_model.process_predictiton(
            model=feature_extraction_para_model,
            **feature_extraction_process_kwargs
        )

        # model_training_model
        prediction_result_df = pd.DataFrame()
        for label_extraction_result in label_extraction_result_list:
            product_id = label_extraction_result['product_id']
            label_result_df = label_extraction_result['label_result_df']
            self.logger.info('product_id:{} begin predict'.format(product_id))

            # model_training_model
            if product_id:
                sub_model_dir = product_id.replace('/', '_')
            else:
                sub_model_dir = None
            model_training_model = self.load_model(
                model_file_name=self.model_training_model_file,
                base_model_dir=self.base_project_models_dir,
                sub_model_dir=sub_model_dir
            )
            if model_training_model is None:
                self.logger.warning('model_file dir: {model_training_model_file} do not have any model'.format(
                    model_training_model_file=self.model_training_model_file
                ))
                continue

            prediction_result_df_tmp = model_training_model.predict(
                feature_result_df=feature_result_df,
                label_result_df=label_result_df,
                logger=self.logger
            )
            # 新增一列
            if product_id is not None:
                prediction_result_df_tmp[label_extraction_model.product_column] = product_id
            prediction_result_df = prediction_result_df.append(prediction_result_df_tmp)

        return prediction_result_df

    def process(self):
        prediction_result_df = self.process_base()
        save_result = self.save_result(prediction_result_df)
        api_result = self.build_api_result(save_result)
        return api_result

    def save_result(self, build_result):
        # 保存
        if build_result is not None:
            prediction_result_url = self.default_prediction_result_url
            build_result.to_csv(prediction_result_url, index=False)
            save_result = {'prediction_result_url': prediction_result_url}
        else:
            save_result = {'error': 'No result'}

        return save_result

    def build_api_result(self, save_result):
        # 保存
        save_result.update({
            'prediction_id': self.prediction_id,
            'prediction_status': 2
        })

        api_result = save_result
        return api_result
