# coding: utf-8
from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd


# TODO 增加参数自动最优匹配
class CatBoost:
    def __init__(
            self,
            scale_pos_weight,
            label_type,
            thread_count=None,
            depth=5,
            learning_rate=0.01,
            reg_lambda=10,
            iterations=200,
            one_hot_max_size=10,
            rsm=0.7,
            random_seed=1,
            early_stopping_rounds=200,
            metric_period=10,
            use_best_model=True,
            object_features_columns=None,
            regressor_loss_function='RMSE',
            regressor_eval_metric='RMSE',
            classifier_loss_function='Logloss',
            classifier_eval_metric='AUC',
    ):

        self.depth = depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.iterations = iterations
        self.one_hot_max_size = one_hot_max_size
        self.rsm = rsm
        self.random_seed = random_seed
        self.early_stopping_rounds = early_stopping_rounds
        self.metric_period = metric_period
        self.use_best_model = use_best_model
        self.object_features_columns = object_features_columns
        self.regressor_loss_function = regressor_loss_function
        self.regressor_eval_metric = regressor_eval_metric
        self.classifier_loss_function = classifier_loss_function
        self.classifier_eval_metric = classifier_eval_metric
        self.label_type = label_type
        self.scale_pos_weight = scale_pos_weight
        self.thread_count = thread_count
        self.model = None

    def _init_model(self):
        if self.label_type == 'regression':
            base_model = CatBoostRegressor
            loss_function = self.regressor_loss_function
            eval_metric = self.regressor_eval_metric

            model = base_model(
                depth=self.depth,
                learning_rate=self.learning_rate,
                reg_lambda=self.reg_lambda,
                one_hot_max_size=self.one_hot_max_size,
                iterations=self.iterations,
                loss_function=loss_function,
                eval_metric=eval_metric,
                rsm=self.rsm,
                thread_count=self.thread_count,
                # scale_pos_weight=self.scale_pos_weight,
                verbose=True,
                random_seed=self.random_seed
            )

        else:
            base_model = CatBoostClassifier
            loss_function = self.classifier_loss_function
            eval_metric = self.classifier_eval_metric

            model = base_model(
                depth=self.depth,
                learning_rate=self.learning_rate,
                reg_lambda=self.reg_lambda,
                one_hot_max_size=self.one_hot_max_size,
                iterations=self.iterations,
                loss_function=loss_function,
                eval_metric=eval_metric,
                rsm=self.rsm,
                scale_pos_weight=self.scale_pos_weight,
                verbose=True,
                random_seed=self.random_seed
            )
        return model

    def _calc_model_para(self):
        pass

    # def predict(self, x_matrix):
    #     if self.label_type == 'regression':
    #         pred = self.model.predict_proba(x_matrix)[:, 1]
    #     else:
    #         pred = self.model.predict(x_matrix)
    #     return pred

    def fit(self,
            x_train,
            x_test,
            y_train,
            y_test):

        self._calc_model_para()
        self.model = self._init_model()
        eval_set = [(x_test, y_test)]
        categorical_features_indices = self._para_categorical_features_indices(x_train)

        self.model.fit(
            X=x_train,
            y=y_train,
            verbose=True,
            plot=False,
            cat_features=categorical_features_indices,
            eval_set=eval_set,  # 为了提前退出，防止过拟合
            metric_period=self.metric_period,  # 为了提前退出，防止过拟合
            early_stopping_rounds=self.early_stopping_rounds,  # 为了提前退出，防止过拟合
            use_best_model=self.use_best_model  # Can be used only with eval_set. metric_period会失效
        )

    def get_feature_importance_df(
            self,
            type='PredictionValuesChange'
    ):
        feature_importance = pd.DataFrame(
            self.model.get_feature_importance(type=type, prettified=True))
        feature_importance.columns = ['feature_name', 'feature_importance']
        return feature_importance

    def _para_categorical_features_indices(self, feature_df):
        if self.object_features_columns is None:
            categorical_features_indices = [i for i, type in enumerate(feature_df.dtypes) if type == 'object']
            self.object_features_columns = feature_df.iloc[:, categorical_features_indices].columns
        else:
            categorical_features_indices = [i for i, column in enumerate(feature_df.columns)
                                            if column in self.object_features_columns]
        return categorical_features_indices
