# coding: utf-8
import logging
logging.basicConfig(level=logging.DEBUG)


class FeatureEngineering:
    def __init__(
            self,
            str_id_col_thres_num=1000,
            str_id_col_threshold=1,
            int_id_col_threshold=0.9,
            default_time_col_name='time',
            logger=None
    ):
        self.str_id_col_thres_num = str_id_col_thres_num
        self.str_id_col_threshold = str_id_col_threshold
        self.int_id_col_threshold = int_id_col_threshold
        self.default_time_col_name = default_time_col_name
        self.logger = logger

        self.fit_str_id_col_l = []
        self.fit_int_id_col_l = []
        self.del_special_features = []

    def fit(self,
            feature_result_df,
            keep_id_l,
            id_series,
            project_id=None
            ):
        if feature_result_df.shape[0] == 0:
            return feature_result_df

        feature_result_df = self._fit_del_str_id_features(
            feature_result_df=feature_result_df,
            keep_id_l=keep_id_l
        )
        # feature_result_df = self._fit_del_int_id_features(
        #     feature_result_df=feature_result_df,
        #     id_series=id_series,
        #     keep_id_l=keep_id_l
        # )
        feature_result_df = self._fit_del_special_features(
            feature_result_df=feature_result_df,
            project_id=project_id,
            keep_id_l=keep_id_l
        )

        return feature_result_df

    def predict(self, feature_result_df):
        if feature_result_df.shape[0] == 0:
            return feature_result_df

        feature_result_df = self._predict_del_str_id_features(feature_result_df)
        # feature_result_df = self._predict_del_int_id_features(feature_result_df)
        feature_result_df = self._predict_del_special_features(feature_result_df)
        return feature_result_df

    def _fit_del_str_id_features(self, feature_result_df, keep_id_l=[]):
        # object id
        str_id_col_thres_num = min(feature_result_df.shape[0]*self.str_id_col_threshold,  self.str_id_col_thres_num)
        object_df = feature_result_df.select_dtypes(include=[object])
        if object_df.shape[0] == 0:
            return feature_result_df

        str_id_col = object_df.columns[object_df.nunique() > str_id_col_thres_num].tolist()
        self.fit_str_id_col_l = list(set(str_id_col) - set(keep_id_l))
        feature_result_df.drop(columns=self.fit_str_id_col_l, inplace=True)
        logging.warning('###################################FeatureEngineering._fit_del_str_id_features del{}'.format(self.fit_str_id_col_l))
        return feature_result_df

    def _fit_del_int_id_features(self, feature_result_df, id_series, keep_id_l=[]):
        # object id
        id_nunique = id_series.nunique()
        int_id_col_thres_num = id_nunique*self.int_id_col_threshold
        int_df = feature_result_df.select_dtypes(include=[int])
        if int_df.shape[0] == 0:
            return feature_result_df

        int_id_col = int_df.columns[int_df.nunique() > int_id_col_thres_num].tolist()
        self.fit_int_id_col_l = list(set(int_id_col) - set(keep_id_l))
        feature_result_df.drop(columns=self.fit_int_id_col_l, inplace=True)
        logging.warning('###################################FeatureEngineering.fit_int_id_col_l del{}'.format(self.fit_int_id_col_l))
        return feature_result_df

    def _fit_del_special_features(self, feature_result_df, project_id, keep_id_l=[]):
        return feature_result_df

    def _predict_del_str_id_features(self, feature_result_df):
        # ValueError: Need to specify at least one of 'labels', 'index' or 'columns'
        feature_result_df.drop(columns=self.fit_str_id_col_l, inplace=True)
        return feature_result_df

    def _predict_del_int_id_features(self, feature_result_df):
        feature_result_df.drop(columns=self.fit_int_id_col_l, inplace=True)
        return feature_result_df

    def _predict_del_special_features(self, feature_result_df):
        feature_result_df.drop(columns=self.del_special_features, inplace=True)
        return feature_result_df

