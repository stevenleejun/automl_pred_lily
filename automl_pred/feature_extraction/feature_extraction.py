# coding: utf-8
import sys
from featuretools.primitives import make_agg_primitive
from featuretools.variable_types import DatetimeTimeIndex, Numeric
import featuretools as ft
import pandas as pd
from ..entity_set_build.entity_set_build import EntitySetBuild
from .feature_engineering import FeatureEngineering
import logging
logging.basicConfig(level=logging.DEBUG)


class FeatureExtractionPara:
    _timedelta_mapper = {'months': 'mo', 'days': 'd'}

    def __init__(
            self,
            customer_entity,
            customer_entity_index,
            training_window_unit,
            training_window,
            agg_primitives=None,
            trans_primitives=None,
            ignore_entities=None,
            ignore_variables=None,
            n_jobs=1,
            chunk_size=0.1,
            drop_contains=None,
            drop_exact=None,
            entity_set_drop_index_list=[],
            auto_max_values=None,
            manual_interesting_values_info=None,
            where_primitives=None,
            default_time_col_name='time',
            str_id_col_threshold=0.9

    ):
        self.customer_entity = customer_entity
        self.customer_entity_index = customer_entity_index
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives
        self.ignore_entities = ignore_entities
        self.ignore_variables = ignore_variables
        self.training_window_unit = training_window_unit
        self.training_window = training_window
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size
        self.drop_contains = drop_contains
        self.drop_exact = drop_exact
        self.entity_set_drop_index_list = entity_set_drop_index_list
        self.auto_max_values = auto_max_values
        self.manual_interesting_values_info = manual_interesting_values_info
        self.where_primitives = where_primitives
        self.default_time_col_name = default_time_col_name

        self.str_id_col_threshold = str_id_col_threshold

        self.feature_engineering_class = self.get_feature_engineering_class()

        # 进一步处理
        self.training_window = ft.Timedelta(
            self.training_window,
            unit=self._timedelta_mapper[self.training_window_unit])

        if self.drop_contains is None:
            self.drop_contains = entity_set_drop_index_list
        else:
            self.drop_contains += entity_set_drop_index_list


        # if self.ignore_variables is None:
        #     self.ignore_variables = {self.customer_entity: self.customer_entity_index}
        # elif self.ignore_variables.get(self.customer_entity, None) is None:
        #     self.ignore_variables[self.customer_entity] = [self.customer_entity_index]
        # else:
        #     self.ignore_variables[self.customer_entity].append(self.customer_entity_index)
    @property
    def get_customer_entity_index(self):
        return self.customer_entity_index

    @staticmethod
    def stat_result(feature_result_df):
        feature_statistics_list = feature_result_df.columns.tolist()
        feature_statistics_df = pd.DataFrame({
            'index': range(len(feature_statistics_list)),
            'feature_name': feature_statistics_list
        })
        return feature_statistics_df

    def get_feature_engineering_class(self):
        feature_engineering = FeatureEngineering(
            str_id_col_threshold=self.str_id_col_threshold,
        )
        return feature_engineering

    def process_feature_engineering(self, is_train, feature_result_df, project_id=None):
        # 特征工程
        if is_train:
            keep_id_l = [self.customer_entity_index, self.default_time_col_name]
            id_series = feature_result_df[self.customer_entity_index]
            feature_result_df = self.feature_engineering_class.fit(
                feature_result_df=feature_result_df,
                keep_id_l=keep_id_l,
                id_series=id_series,
                project_id=project_id
            )
        else:
            feature_result_df = self.feature_engineering_class.predict(feature_result_df)

        return feature_result_df

    def reset_feature_engineering_class(self, feature_engineering_class):
        self.feature_engineering_class = feature_engineering_class
        logging.info(self.feature_engineering_class.fit_str_id_col_l)


class FeatureExtraction:
    def __init__(
            self,
            feature_extraction_para
    ):
        self.customer_entity = feature_extraction_para.customer_entity
        self.customer_entity_index = feature_extraction_para.customer_entity_index
        self.agg_primitives = feature_extraction_para.agg_primitives.copy()
        self.trans_primitives = feature_extraction_para.trans_primitives.copy()
        self.ignore_entities = feature_extraction_para.ignore_entities
        self.ignore_variables = feature_extraction_para.ignore_variables
        self.training_window_unit = feature_extraction_para.training_window_unit
        self.training_window = feature_extraction_para.training_window
        self.n_jobs = feature_extraction_para.n_jobs
        self.chunk_size = feature_extraction_para.chunk_size
        self.drop_contains = feature_extraction_para.drop_contains
        self.drop_exact = feature_extraction_para.drop_exact
        self.auto_max_values = feature_extraction_para.auto_max_values
        self.manual_interesting_values_info = feature_extraction_para.manual_interesting_values_info
        self.where_primitives = feature_extraction_para.where_primitives
        self.default_time_col_name = feature_extraction_para.default_time_col_name

        self.feature_engineering_class = feature_extraction_para.feature_engineering_class

        self.base_model = feature_extraction_para

        self._get_agg_primitives()

    def transform(self, entity_set, cutoff_time, logger, is_train):
        # 读入label_result_df
        logger.info("读入label_result_df begin")
        if cutoff_time.shape[0] == 0:
            return pd.DataFrame()

        cutoff_time[self.default_time_col_name] = pd.to_datetime(cutoff_time[self.default_time_col_name])
        cutoff_time = cutoff_time[[self.customer_entity_index, self.default_time_col_name]]

        # 创建 FeatureExtraction
        logger.info("创建 feature_result_df begin")

        entity_set = self._rebuild_entity_set(entity_set)

        #TODO
        if 'n_most_common' in self.agg_primitives:
            self.agg_primitives.remove('n_most_common')

        feature_result_dfs = ft.dfs(
            entityset=entity_set,
            target_entity=self.customer_entity,
            cutoff_time=cutoff_time,
            agg_primitives=self.agg_primitives,
            trans_primitives=self.trans_primitives,
            ignore_entities=self.ignore_entities,
            ignore_variables=self.ignore_variables,
            training_window=self.training_window,
            cutoff_time_in_index=True,
            features_only=False,
            chunk_size=self.chunk_size,
            drop_contains=self.drop_contains,
            drop_exact=self.drop_exact,
            verbose=True,
            where_primitives=self.where_primitives,
            return_variable_types='all'
        )

        # 美化log文件
        sys.stdout.write('')
        sys.stdout.flush()

        if isinstance(feature_result_dfs, tuple):
            feature_result_df, _ = feature_result_dfs
        else:
            feature_result_df = pd.DataFrame(columns=list(map(str, feature_result_dfs)))
        feature_result_df.reset_index(inplace=True)

        return feature_result_df

    def refresh_base_model(self):
        self.base_model.reset_feature_engineering_class(self.feature_engineering_class)
        return self.base_model

    def _rebuild_entity_set(self, entity_set):
        entity_set_build = EntitySetBuild(
            auto_max_values=self.auto_max_values,
            manual_interesting_values_info=self.manual_interesting_values_info
        )
        entity_set = entity_set_build.build_interesting_value(entity_set)
        return entity_set

    def _make_agg_primitives(self):
        self.days_since_last = make_agg_primitive(
            function=self._days_since_last,
            name='days_since_last',
            input_types=[DatetimeTimeIndex],
            return_type=Numeric,
            description="Time since last related instance",
            uses_calc_time=True
        )
        self.month_of_cutoff_point = make_agg_primitive(
            function=self._month_of_cutoff_point,
            name='month_of_cutoff_point',
            input_types=[DatetimeTimeIndex],
            return_type=Numeric,
            description="month_of_cutoff_point",
            uses_calc_time=True
        )
        self.user_defined_agg_primitives = ['month_of_cutoff_point']

    def _get_agg_primitives(self):
        self._make_agg_primitives()
        for primitive in self.user_defined_agg_primitives:
            if primitive in self.agg_primitives:
                self.agg_primitives.remove(primitive)
        self.agg_primitives.append(self.month_of_cutoff_point)

    def _month_of_cutoff_point(self, tmp, time):
        # Specify the inputs and return
        return time.month

    def _days_since_last(self, values, time):
        # 因为少了一个self，改bug改了一下午
        time_since = time - values.iloc[-1]
        result = time_since.days
        return result

    # def month_of_cutoff_point(self, time):
    #     # 因为少了一个self，改bug改了一下午
    #     time_since = time - values.iloc[-1]
    #     result = time_since.days
    #     return result

    # values =\
    #     func(args[0], time=time_last)
    #
    # values = args[0]
    # func(1, time=time_last)
    # days_since_last_fun(args[0], time=time_last)
    #
