import os
import pandas as pd
import sys

from .base_process import BaseProcess
from ..model_training.model_training import ModelTraining


class ModelTrainingProcess(BaseProcess):
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
        # project_id
        self.model_training_id = self.process_config.get('model_training_id', 'no_model_training_id')
        self.feature_result_url = self.process_config.get('feature_result_url', self.default_feature_result_url)
        self.label_url_list = self.process_config.get('label_url_list', self.default_label_url_list)

    def init_main_class(self):
        pass

    def process_base(self, model=None, **kwargs):
        try:
            # feature_result_df = pd.read_csv(self.feature_result_url)
            # feature_result_df = pd.read_pickle(self.feature_result_url)
            feature_result_df = pd.read_hdf(self.feature_result_url)
        except:
            self.logger.error('feature_result_df = pd.read_hdf(self.feature_result_url)')
            return []

        model_info_dic_list = []
        # 根据标签结果列表
        for label_url in self.label_url_list:
            product_id = label_url.get('product_id', None)
            label_result_url = label_url['label_result_url']

            # 模型目录
            if product_id:
                self.logger.info('{}:  begin'.format(product_id))
                project_models_dir = os.path.join(self.base_project_models_dir, product_id.replace('/', '_'))
                if not os.path.exists(project_models_dir):
                    os.makedirs(project_models_dir)
            else:
                project_models_dir = self.base_project_models_dir

            # 读取 label_result_df
            if not os.path.exists(label_result_url):
                self.logger.error("not os.path.exists(label_result_url)")
                continue
            else:
                label_result_df = pd.read_csv(label_result_url)

            # 创建 ModelTraining
            model_training_model = ModelTraining(
                problem_type=self.problem_type,
                label_type=self.label_type,
                thread_count=self.num_workers
            )

            x_train, x_test, y_train, y_test = model_training_model.fit(
                feature_result_df=feature_result_df,
                label_result_df=label_result_df,
                logger=self.logger
            )

            if x_train is None:
                continue

            self.save_model(model_training_model, self.model_training_model_file, project_models_dir)

            model_info_dic = self.__build_single_model_result(
                product_id,
                x_train,
                x_test,
                y_train,
                y_test,
                project_models_dir,
                model_training_model
            )
            model_info_dic_list.append(model_info_dic)
        return model_info_dic_list
    
    def process(self):
        self._copy_before_model_file()
        model_info_dic_list = self.process_base()
        result = self.build_result(model_info_dic_list)
        return result

    def build_result(self, model_info_dic_list):
        result = {}
        result['model_training_id'] = self.model_training_id
        result['base_model_result_dir'] = self.base_project_models_dir
        result['model_list'] = model_info_dic_list
        result['training_status'] = 2
        
        return result

    def __build_single_model_result(
            self,
            product_id,
            x_train,
            x_test,
            y_train,
            y_test,
            project_models_dir,
            model_training_model
    ):
        train_evaluate_dic = model_training_model.evaluate(
            x_train,
            y_train,
            project_models_dir,
            file_prefix='train',
            logger=self.logger
        )

        test_evaluate_dic = model_training_model.evaluate(
            x_test,
            y_test,
            project_models_dir,
            file_prefix='test',
            logger=self.logger
        )

        feature_importance_path = model_training_model.feature_importance(project_models_dir)

        model_info_dic = {}
        # 模型id
        model_info_dic['product_id'] = product_id
        model_info_dic['model_result_dir'] = project_models_dir

        # 模型描述信息
        model_info_dic['training_samples_number_desc'] = \
            format(model_training_model.training_samples_number, ',') \
            + '，平均每个观测点样本数' \
            + format(model_training_model.training_samples_number / model_training_model.cutoff_num, ',.0f')
        model_info_dic['features_number_desc'] = str(model_training_model.feature_num) + '个'
        if self.label_type == 'classifier':
            model_info_dic['training_label_desc'] = 'Label=1占比' + format(model_training_model.pos_rate, '.0%')
        else:
            model_info_dic['training_label_desc'] = '均值:' + format(model_training_model.y_mean, ',.2f') \
                                                 + '中位数:' + format(model_training_model.y_median, ',.2f')
        
        # 模型评估信息
        model_evaluation_list = self.__build_result_evaluate_dic(
            evaluate_dic=train_evaluate_dic,
            file_prefix='train'
        )
        model_evaluation_list_test = self.__build_result_evaluate_dic(
            evaluate_dic=test_evaluate_dic,
            file_prefix='test'
        )
        model_evaluation_list.extend(model_evaluation_list_test)
        
        # feature_importance_path
        model_evaluation_list.append({
            'value_url': feature_importance_path,
            'model_split_key': 3,
            'model_eval_key': 5
        })
        
        model_info_dic['model_evaluation'] = model_evaluation_list
        
        return model_info_dic
    
    def __build_result_evaluate_dic(self, evaluate_dic, file_prefix):
        model_evaluation_list = []
        model_split_key = self.model_split_key_value[self.label_type][file_prefix]
        for eval_key, eval_value in self.model_eval_key_value[self.label_type].items():
            model_evaluation_list.append({
                'value_url': evaluate_dic.get(eval_key, None),
                'model_split_key': model_split_key,
                'model_eval_key': eval_value
            })
        return model_evaluation_list
