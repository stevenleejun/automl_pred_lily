# coding: utf-8
import sys
from sklearn.model_selection import train_test_split

import scikitplot as skplt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


from utils.utils_model_ana import model_profit_report_classifier
from utils.utils_model_ana import plot_feature_importance

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from utils.utils_model_ana import plot_abs_error_recall
from utils.utils_model_ana import plot_cumulative_gains_regression
from utils.utils_model_ana import plot_pred_vs_true
from utils.utils_model_ana import plot_abs_percent_error_recall
from utils.utils_model_ana import model_profit_report_regression

from utils.utils_df import get_df_columns_by_data_type

from .cat_boost import CatBoost
import warnings
import logging
import random
import dask.dataframe as dd
from dask.distributed import Client
from dask.distributed import LocalCluster
import multiprocessing as mp
import matplotlib

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  #用来正常显示中文标签
matplotlib.use('Agg')
warnings.filterwarnings('ignore')


class ModelTraining:
    def __init__(
            self,
            problem_type,
            label_type,
            model_class=CatBoost,
            if_log_transform=1,
            test_size=0.3,
            random_state=1,
            scale_pos_weight_threshold=10,
            object_features_columns=[],
            thread_count=None,
            default_time_col_name='time',
            train_test_split_type='split_by_id',
            prediction_col_name='prediction',
            cutoff_col_name='cutoff',
            label_col_name='label',
            decile_col_name='decile',
            **kwargs
    ):
        self.model_class = model_class
        self.problem_type = problem_type
        self.label_type = label_type
        self.test_size = test_size
        self.if_log_transform = if_log_transform
        self.random_state = random_state
        self.scale_pos_weight_threshold = scale_pos_weight_threshold
        self.thread_count = thread_count
        self.default_time_col_name = default_time_col_name
        self.train_test_split_type = train_test_split_type

        self.prediction_col_name = prediction_col_name
        self.cutoff_col_name = cutoff_col_name
        self.label_col_name = label_col_name
        self.decile_col_name = decile_col_name

        self.kwargs = kwargs

        self.object_features_columns = object_features_columns
        self.int_features_columns = []
        self.float_features_columns = []
        self.datetime_features_columns = []
        self.bool_features_columns = []

        self.model = None
        self.model_feature_columns = None
        self.training_samples_number = None
        self.cutoff_num = None
        self.feature_num = None

        self.pos_rate = None
        self.scale_pos_weight = None
        self.threshold = None

        self.y_mean = None
        self.y_median = None

        self.customer_entity_index = None

    def fit(self, feature_result_df, label_result_df, logger):
        logger.info('ModelTraining fit begin.')
        if (feature_result_df.shape[0] == 0) or (label_result_df.shape[0] == 0):
            logger.error('(feature_result_df.shape[0] == 0) or (label_result_df.shape[0] == 0)')
            return None, None, None, None

        feature_df, label_df, id_df, cutoff_time_df = self._merge_feature_and_label(
            feature_result_df,
            label_result_df
        )

        stratify = self._para_stratify(label_df)
        x_train, x_test, y_train, y_test = self._train_test_split(
            feature_df=feature_df,
            label_df=label_df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
            id_df=id_df
        )
        label_df = label_df[self.problem_type]

        if self.label_type == 'classifier':
            if label_df.nunique() == 1:
                logger.error('model_carpartin_train:{}, train_data_df[self.problem_type].nunique()==1')
                return None, None, None, None

            if label_df.value_counts().min() == 1:
                logger.error('The least populated class in y has only 1 member, which is too few.')
                return None, None, None, None

        if y_train.nunique() == 1:
            logger.error('model_carpartin_train:{}, train_data_df[self.problem_type].nunique()==1')
            return None, None, None, None

        self._self_scale_pos_weight(label_df)

        # model
        best_model = self.model_class(
            label_type=self.label_type,
            scale_pos_weight=self.scale_pos_weight,
            thread_count=self.thread_count,
            object_features_columns=self.object_features_columns,
            **self.kwargs
        )
        best_model.fit(
            x_train,
            x_test,
            y_train,
            y_test
        )

        # 美化log文件
        sys.stdout.write('')
        sys.stdout.flush()

        # model信息储存
        self.model = best_model
        self.model_feature_columns = feature_df.columns.values

        self.training_samples_number = feature_df.shape[0]
        self.cutoff_num = cutoff_time_df.nunique()
        self.feature_num = feature_df.shape[1]

        if self.label_type == 'classifier':
            y_pred = self.model.model.predict_proba(x_train)
            self.pos_rate = label_df[label_df == 1].shape[0] / label_df.shape[0]
            self.threshold = np.quantile(y_pred[:, -1], q=1 - self.pos_rate)

        else:
            self.y_mean = label_df.mean()
            self.y_median = label_df.median()

        return x_train, x_test, y_train, y_test

    def predict(
            self,
            feature_result_df,
            label_result_df,
            logger
    ):
        if (feature_result_df.shape[0] == 0) or (label_result_df.shape[0] == 0):
            return pd.DataFrame()

        feature_df, label_df, id_df, cutoff_time_df = self._merge_feature_and_label(
            feature_result_df,
            label_result_df,
            is_train=False
        )

        # 预测列少的时候要补null
        more_model_col = [model_col for model_col in self.model_feature_columns if model_col not in feature_df.columns]
        for more_col in more_model_col:
            feature_df[more_col] = np.NaN

        feature_df = feature_df[self.model_feature_columns]
        if self.label_type == 'classifier':
            y_prediction = self.model.model.predict_proba(feature_df)[:, 1]
        else:
            y_prediction = self.model.model.predict(feature_df)

        predict_df = pd.DataFrame({
            id_df.name: id_df,
            self.cutoff_col_name: cutoff_time_df,
            self.prediction_col_name: y_prediction
        }).sort_values(self.prediction_col_name, ascending=False)

        if self.label_type == 'classifier':
            predict_df = self._rebuild_predict_df_for_classifier(
                predict_df=predict_df,
                true_threshold=self.threshold
            )

        return predict_df

    def feature_importance(self, project_models_dir, type='PredictionValuesChange'):
        # 特征重要性
        feature_importance = self.model.get_feature_importance_df()

        title = 'feature_importance'
        plot_feature_importance(feature_importance, type=type)
        feature_importance_path = os.path.join(project_models_dir, title + ".png")
        plt.savefig(feature_importance_path, bbox_inches="tight")
        feature_importance.to_csv(os.path.join(project_models_dir, title + ".csv"), encoding="utf-8", index=True)

        return feature_importance_path

    def evaluate(self,
                 x_true,
                 y_true,
                 project_models_dir,
                 file_prefix='',
                 logger=None
                 ):
        if self.label_type == 'classifier':
            evaluate_fun = self._evaluate_for_classifier
        else:
            evaluate_fun = self._evaluate_for_regression

        evaluate_dic = evaluate_fun(
            x_true=x_true,
            y_true=y_true,
            project_models_dir=project_models_dir,
            file_prefix=file_prefix,
            logger=logger
        )
        return evaluate_dic

    def _train_test_split(
            self,
            feature_df,
            label_df,
            test_size,
            random_state,
            stratify,
            id_df
    ):
        if self.train_test_split_type == 'split_by_last_time':
            is_test_index = label_df.index.get_level_values(self.default_time_col_name) ==\
                            label_df.index.get_level_values(self.default_time_col_name).max()
            is_train_index = list(map(lambda x: not x, is_test_index))

            x_train = feature_df.loc[is_train_index, :]
            x_test = feature_df.loc[is_test_index, :]
            y_train = label_df.loc[is_train_index, :]
            y_test = label_df.loc[is_test_index, :]

        elif self.train_test_split_type == 'split_by_id':
            id_list = label_df.index.get_level_values(self.customer_entity_index).unique().tolist()
            id_list_test = random.sample(id_list, int(len(id_list) * test_size))

            is_test_index = list(map(lambda x: x in id_list_test,
                                     label_df.index.get_level_values(self.customer_entity_index).tolist()
                                     )
                                 )
            is_train_index = list(map(lambda x: not x, is_test_index))

            x_train = feature_df.loc[is_train_index, :]
            x_test = feature_df.loc[is_test_index, :]
            y_train = label_df.loc[is_train_index, :]
            y_test = label_df.loc[is_test_index, :]

        else:
            x_train, x_test, y_train, y_test = train_test_split(
                feature_df,
                label_df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify
            )
        y_train = y_train[self.problem_type]
        y_test = y_test[self.problem_type]
        return x_train, x_test, y_train, y_test

    def _para_stratify(self, label_df):
        if self.label_type == 'classifier':
            stratify = label_df
        else:
            stratify = None

        return stratify

    def _self_scale_pos_weight(self, label_df):
        scale_pos_weight = None
        if self.label_type == 'classifier':
            scale_pos_weight = label_df[label_df == 0].shape[0] / label_df[label_df == 1].shape[0]
            if scale_pos_weight < self.scale_pos_weight_threshold:
                scale_pos_weight = None
        self.scale_pos_weight = scale_pos_weight

    def _merge_feature_and_label(
            self,
            feature_result_df,
            label_result_df,
            is_train=True
    ):
        # 获取 x_train, x_test, y_train, y_test
        feature_result_df[self.default_time_col_name] = pd.to_datetime(feature_result_df[self.default_time_col_name])
        label_result_df[self.default_time_col_name] = pd.to_datetime(label_result_df[self.default_time_col_name])
        
        cutoff_time_column = self.default_time_col_name
        label_column = self.problem_type
        self.customer_entity_index = list(set(label_result_df.columns) - set([cutoff_time_column, label_column]))[0]

        try:
            train_data_df = pd.merge(
                feature_result_df,
                label_result_df,
                on=[self.default_time_col_name, self.customer_entity_index]
            )

        except Exception as ex:
            with LocalCluster(processes=True,
                              threads_per_worker=1,
                              memory_limit='20GB',
                              # ip='tcp://localhost:9895',
                              ) as cluster, Client(cluster) as client:
                client.cluster.scale(int(0.8 * mp.cpu_count()))  # ask for ten 4-thread workers
                # MERGE IN DASK
                logging.warning('=============dd.merge(feature_result_df_dd, label_result_df_dd)  begin')
                feature_result_df_dd = dd.from_pandas(feature_result_df, npartitions=10)
                label_result_df_dd = dd.from_pandas(label_result_df, npartitions=2)
                train_data_df_dd = dd.merge(
                    feature_result_df_dd,
                    label_result_df_dd)
                train_data_df = train_data_df_dd.compute()
                logging.warning('=============feature_result_df:{}, label_result_df:{}, train_data_df:{}'.format(
                    feature_result_df.shape,
                    label_result_df.shape,
                    train_data_df.shape
                ))

        id_df = train_data_df[self.customer_entity_index]
        cutoff_time_df = train_data_df[cutoff_time_column]
        label_df = train_data_df[[label_column, self.customer_entity_index, cutoff_time_column]]
        train_data_df.drop(columns=[label_column], inplace=True)
        feature_df = train_data_df  # .drop(columns=[self.customer_entity_index, cutoff_time_column])

        feature_df.set_index([self.customer_entity_index, cutoff_time_column], inplace=True)
        label_df.set_index([self.customer_entity_index, cutoff_time_column], inplace=True)

        feature_df = self._deal_feature_for_data_type(feature_df)

        return feature_df, label_df, id_df, cutoff_time_df

    def _self_features_columns_for_data_types(self, feature_df):
        if len(self.object_features_columns) == 0:
            self.object_features_columns = get_df_columns_by_data_type(
                data_df=feature_df,
                data_type='object'
            )

        if len(self.int_features_columns) == 0:
            self.int_features_columns = get_df_columns_by_data_type(
                data_df=feature_df,
                data_type='int'
            )

        if len(self.float_features_columns) == 0:
            self.float_features_columns = get_df_columns_by_data_type(
                data_df=feature_df,
                data_type='float'
            )

        if len(self.datetime_features_columns) == 0:
            self.datetime_features_columns = get_df_columns_by_data_type(
                data_df=feature_df,
                data_type='datetime'
            )

        if len(self.bool_features_columns) == 0:
            self.bool_features_columns = get_df_columns_by_data_type(
                data_df=feature_df,
                data_type='bool'
            )

    def _deal_feature_for_data_type(self, feature_df, is_train=True):
        if is_train: #在训练的时候才更新
            self._self_features_columns_for_data_types(feature_df)

        date_cols = self.datetime_features_columns
        logging.debug('begin convert type date_cols:{}'.format(date_cols))
        # for col in date_cols:
        #     if col in feature_df.columns:
        #         feature_df[col] = pd.to_datetime(feature_df[col], errors='coerce')
        #         feature_df[col][pd.isnull(feature_df[col])] = np.NaN  # TypeError: Cannot convert obj NaT to float
        feature_df.drop(columns=self.datetime_features_columns, inplace=True)

        float_cols = self.float_features_columns
        logging.debug('begin convert type float_cols:{}'.format(float_cols))
        for col in float_cols:
            if col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                feature_df[col] = feature_df[col].astype('float')
                feature_df[col][pd.isnull(feature_df[col])] = np.NaN  # TypeError: Cannot convert obj NaT to float

        int_cols = self.int_features_columns
        logging.debug('begin convert type int_cols:{}'.format(int_cols))
        for col in int_cols:
            if col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], downcast='integer', errors='coerce')
                feature_df[col] = feature_df[col].astype('float')  # int值不能有nan
                feature_df[col][pd.isnull(feature_df[col])] = np.NaN  # TypeError: Cannot convert obj NaT to float

        str_cols = self.object_features_columns
        logging.debug('begin convert type str_cols:{}'.format(str_cols))
        for col in str_cols:
            if col in feature_df.columns:
                feature_df[col][pd.isnull(feature_df[col])] = np.NaN  # TypeError: Cannot convert obj NaT to float
                feature_df[col] = feature_df[col].fillna('other')
                feature_df[col] = feature_df[col].astype('str')
                # feature_df[col][feature_df[col] == 'null'] = np.nan

        logging.debug('feature_df.columns[feature_df[pd.isnull(feature_df)].sum()>0]:{}'.format(
            feature_df.columns[feature_df[pd.isnull(feature_df)].sum() > 0]
        ))

        logging.debug("feature_df.columns[feature_df[feature_df == 'NaT']].sum()>0]:{}".format(
            feature_df.columns[feature_df[feature_df == 'NaT'].sum() > 0]
        ))

        logging.debug("feature_df.columns[feature_df[feature_df.astype('str') == 'NaT'].sum() > 0]:{}".format(
            feature_df.columns[feature_df[feature_df.astype('str') == 'NaT'].sum() > 0]
        ))

        feature_df[pd.isnull(feature_df)] = np.NaN # TypeError: Cannot convert obj NaT to float
        feature_df[feature_df == 'NaT'] = np.NaN
        feature_df[feature_df.astype('str') == 'NaT'] = np.NaN

        # 数据类型转化
        logging.debug("feature_df_columns in model save to ../../tmp:{}".format(
            feature_df.columns
        ))
        tmp_dir = '../../tmp_data'
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        pd.DataFrame(feature_df.columns).to_csv(os.path.join(tmp_dir, 'feature_df_columns.csv'), index=True)

        return feature_df

    def _rebuild_predict_df_for_classifier(
            self,
            predict_df,
            threshold_base='true_threshold',
            true_threshold=0.5,
            true_percent=0.3
    ):
        if threshold_base == 'true_threshold':
            threshold = true_threshold
        else:
            threshold = predict_df[self.prediction_col_name].quantile(1 - true_percent)  # 按照prediction前百分之40将pre打成1

        predict_df[self.label_col_name] = 0
        predict_df.loc[predict_df[self.prediction_col_name] >= threshold, self.label_col_name] = 1
        predict_df[self.decile_col_name] = pd.qcut(
            predict_df[self.prediction_col_name].rank(method='first'),
            10,
            labels=list(reversed([i for i in np.arange(1, 11) * 0.1]))
        )
        predict_df[self.prediction_col_name] = predict_df[self.prediction_col_name].round(decimals=4)
        # predict_df['threshould_prediction'] = threshold
        predict_df['train_pos_rate'] = self.pos_rate
        predict_df['train_pos_rate_threshold'] = self.threshold

        return predict_df

    def _evaluate_for_classifier(self,
                 x_true, 
                 y_true, 
                 project_models_dir,
                 file_prefix='',
                 logger=None
                 ):
        if y_true.nunique() == 1:
            logger.error(' File /Users/mjiang/anaconda3/lib/python3.6/site-packages/scikitplot/metrics.py, line 1103, in plot_cumulative_gain {} category/ies.format(len(classes)))')
            return {}
        y_pred = self.model.model.predict_proba(x_true)
        y_pred_class = self.model.model.predict(x_true)
        y_pred_onelevel = y_pred[:, 1]

        # plot_cumulative_gain
        title = file_prefix + '_' + 'cumulative_gain'
        skplt.metrics.plot_cumulative_gain(y_true, y_pred, title=title)
        cumulative_gain_curve_path = os.path.join(project_models_dir, title + ".png")
        plt.savefig(cumulative_gain_curve_path)

        # plot_roc
        title = file_prefix + '_' + 'ROC'
        skplt.metrics.plot_roc(y_true, y_pred, plot_micro=False, plot_macro=False, title=title)
        roc_curve_path = os.path.join(project_models_dir, title + ".png")
        plt.savefig(roc_curve_path)

        # plot_confusion_matrix
        title = file_prefix + '_' + 'confusion_matrix'
        cm = confusion_matrix(y_true, y_pred_class)
        plot_confusion_matrix(
            conf_mat=cm,
            show_absolute=True,
            show_normed=True,
            colorbar=True
        )
        confusion_matrix_path = os.path.join(project_models_dir, title + ".png")
        plt.savefig(confusion_matrix_path, bbox_inches="tight")

        # model_profit_report_classifier
        model_profit_report_classifier_df = model_profit_report_classifier(y_true, y_pred_onelevel)
        title = file_prefix + '_' + 'model_profit_report_classifier_df'
        model_profit_report_classifier_df_path = os.path.join(project_models_dir, title + ".csv")
        model_profit_report_classifier_df.to_csv(model_profit_report_classifier_df_path, index=False)

        # auc
        auc = roc_auc_score(y_true, y_pred_onelevel)

        # 结果整理
        evaluate_dic = {
            'cumulative_gain_curve_path': cumulative_gain_curve_path,
            'roc_curve_path': roc_curve_path,
            'confusion_matrix_path': confusion_matrix_path,
            'model_profit_report_classifier_df_path': model_profit_report_classifier_df_path,
            'auc': format(auc,  ',.2f')
        }

        return evaluate_dic

    def _evaluate_for_regression(self,
                 x_true,
                 y_true,
                 project_models_dir,
                 file_prefix='',
                 logger=None
                 ):
        y_pred = self.model.model.predict(x_true)
        
        # plot_pred_vs_true
        title = file_prefix + '_' + 'pred_vs_true'
        pred_vs_true, sns_plot = plot_pred_vs_true(y_true, y_pred)
        # pred_vs_true = plot_pred_vs_true(y_true, y_pred)
        pred_vs_true_path = os.path.join(project_models_dir, title + ".csv")
        pred_vs_true.to_csv(pred_vs_true_path, index=False)
        pred_vs_true_curve_path = os.path.join(project_models_dir, title + ".png")
        sns_plot.figure.savefig(pred_vs_true_curve_path)
        # plt.savefig(pred_vs_true_curve_path)

        # # plot_abs_error_recall
        # title = file_prefix + '_' + 'abs_error_recall'
        # abs_error_recall = plot_abs_error_recall(y_true, y_pred, errors_show_list=np.linspace(1000, 10000, 30))
        # abs_error_recall_path = os.path.join(project_models_dir, title + ".csv")
        # abs_error_recall.to_csv(abs_error_recall_path, index=False)
        # abs_error_recall_curve_path = os.path.join(project_models_dir, title + ".png")
        # plt.savefig(abs_error_recall_curve_path)

        # # plot_abs_percent_error_recall
        # title = file_prefix + '_' + 'abs_percent_error_recall'
        # abs_percent_error_recall = plot_abs_percent_error_recall(y_true, y_pred, errors_show_list=np.arange(0, 2.1, 0.1))
        # abs_percent_error_recall_path = os.path.join(project_models_dir, title + ".csv")
        # abs_percent_error_recall.to_csv(abs_percent_error_recall_path, index=False)
        # abs_percent_error_recall_curve_path = os.path.join(project_models_dir, title + ".png")
        # plt.savefig(abs_percent_error_recall_curve_path)
        
        # plot_cumulative_gains_regression
        title = file_prefix + '_' + 'cumulative_gain'
        cumulative_gain = plot_cumulative_gains_regression(y_true, y_pred)
        cumulative_gain_path = os.path.join(project_models_dir, title + ".csv")
        cumulative_gain.to_csv(cumulative_gain_path, index=False)
        cumulative_gain_curve_path = os.path.join(project_models_dir, title + ".png")
        plt.savefig(cumulative_gain_curve_path)

        # model_profit_report_regression_df_path
        title = file_prefix + '_' + 'model_profit_report_regression'
        model_profit_report_regression_df = model_profit_report_regression(y_true, y_pred)
        model_profit_report_regression_df_path = os.path.join(project_models_dir, title + ".csv")
        model_profit_report_regression_df.to_csv(model_profit_report_regression_df_path, index=False)
        logging.debug('======================')

        # 定量评估
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        # 结果整理
        evaluate_dic = {
            'abs_error_recall_curve_path': None,
            'abs_percent_error_recall_curve_path': None,
            'cumulative_gain_curve_path': cumulative_gain_curve_path,
            'model_profit_report_regression_df_path': model_profit_report_regression_df_path,
            'pred_vs_true_curve_path': pred_vs_true_curve_path,
            'r2': format(r2, ',.2f'),
            'mae': format(mae, ',.2f'),
            'mse': format(mse, ',.2f')
        }

        return evaluate_dic
