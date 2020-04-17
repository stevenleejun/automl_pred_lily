# -*- coding: utf-8 -*-

from utils.utils_api_base_resource import BaseResource
from webargs import fields, validate

class EntitySetPreviewResource(BaseResource):
    pass
entity_set_preview_parser_args = {
    "entity_set_desc": fields.List(fields.Nested({
        "entity_name": fields.Str(required=True, validate=lambda p: len(p) >= 1),
        "table_name": fields.Str(required=False),
        "sql": fields.Str(allow_none=False),
        "url": fields.Str(allow_none=True),
        "url_sep": fields.Str(allow_none=True),
        "index": fields.List(fields.String, allow_none=True),
        "time_index": fields.Str(allow_none=True),
        "parent_entity": fields.List(fields.Nested({
            "entity_name": fields.Str(allow_none=False),
            "join_column": fields.Str(allow_none=True)
        }, required=False), required=False),
        "normalize_base_entity": fields.Str(allow_none=True),
        "db_info": fields.Nested({
            "type": fields.Str(allow_none=False),
            "host": fields.Str(allow_none=False),
            "port": fields.Int(allow_none=False),
            "user": fields.Str(allow_none=True),
            "password": fields.Str(allow_none=True),
            "database": fields.Str(allow_none=False),
            "extra_para": fields.List(fields.Nested({
                "para_name": fields.Str(allow_none=False),
                "para_value": fields.Str(allow_none=False),
                "para_value_type": fields.Str(allow_none=False),
            }, required=False), required=False),
        }, required=False),
        "field_list": fields.List(fields.Nested({
            "name": fields.Str(allow_none=False),
            "variable_type": fields.Str(allow_none=False)
        }, required=False), required=False),
    }, required=True), required=True)
}


class LabelExtractionResource(BaseResource):
    pass
label_extraction_parser_args = {
    "project_id": fields.Str(required=True),
    "label_extraction_id": fields.Str(required=True),
    "entity_set_desc": entity_set_preview_parser_args["entity_set_desc"],
    "business_desc": fields.Nested({
        "problem_type": fields.Str(required=True),
        "customer_entity": fields.Str(required=True),
        "behavior_entity": fields.Str(required=True),
        "product_entity": fields.Str(allow_none=True),
        "target_definition": fields.Nested({
            "action": fields.List(fields.Nested({
                "column": fields.Str(required=True),
                "operator": fields.Str(required=True),
                "value": fields.Int(required=True),
                "agg": fields.Str(allow_none=True),
            }), allow_none=True),
            "churn_para": fields.Nested({
                "churn_window_unit": fields.Str(required=True),
                "churn_window": fields.Int(required=True)
            }, allow_none=True),
            "product_purchase_para": fields.Nested({
                "product_column": fields.Str(required=True),
                "product_list": fields.List(fields.String, required=True)
            }, allow_none=True),
            "value_para": fields.Nested({
                "value_column": fields.Str(required=True),
                "value_agg": fields.Str(required=True)
            }, allow_none=True),
            "highvalue_para": fields.Nested({
                "highvalue_window_unit": fields.Str(required=True),
                "highvalue_window": fields.Int(required=True)
            }, allow_none=True)
        }, allow_none=True),
    }, required=True),
    "label_desc": fields.Nested({
        "cutoff_point": fields.Nested({
            "cutoff_day": fields.Int(required=True),
            "cutoff_num": fields.Int(required=True),
            "cutoff_last_date": fields.Date(allow_none=True),
        },required=True),
        "lead_time_window": fields.Nested({
            "lead_time_window_unit": fields.Str(required=True),
            "lead_time_window": fields.Int(required=True)
        }, required=True),
        "prediction_window": fields.Nested({
            "prediction_window_unit": fields.Str(required=True),
            "prediction_window": fields.Int(required=True)
        }, required=True),
        "sample_filtering_window": fields.Nested({
            "sample_filtering_window_unit": fields.Str(required=True),
            "sample_filtering_window": fields.Int(required=True)
        }, required=True),
    }, required=True)
}

class FeatureExtractionResource(BaseResource):
    pass
feature_extraction_parser_args = {
    "project_id": fields.Str(required=True),
    "feature_extraction_id": fields.Str(required=True),
    "label_url_list": fields.List(fields.Nested({
        "product_id": fields.Str(allow_none=True),
        "label_result_url": fields.Str(required=True),
    }), required=True),
    "entity_set_desc": entity_set_preview_parser_args['entity_set_desc'],
    "business_desc": fields.Nested({
        "problem_type": fields.Str(allow_none=True),
        "customer_entity": fields.Str(required=True),
        "product_entity": fields.Str(allow_none=True),
        "target_definition": fields.Nested({
            "action": fields.List(fields.Nested({
                "column": fields.Str(required=True),
                "operator": fields.Str(required=True),
                "value": fields.Int(required=True),
                "agg": fields.Str(allow_none=True),
            }), allow_none=True),
            "product_purchase_para": fields.Nested({
                "product_column": fields.Str(required=True),
                "product_list": fields.List(fields.String, required=True)
            }, allow_none=True),
        }, allow_none=True),
    }, required=True),
    "label_desc": fields.Nested({
        "training_window": fields.Nested({
            "training_window_unit": fields.Str(required=True),
            "training_window": fields.Int(required=True)
        }, required=True),
    }, required=True),
    "feature_extraction_desc": fields.Nested({
        "agg_primitives": fields.List(fields.String, allow_none=True),
        "trans_primitives": fields.List(fields.String, allow_none=True),
        "ignore_entities": fields.List(fields.String, allow_none=True),
        "ignore_variables": fields.List(fields.Nested({
                "entity_name": fields.Str(required=True),
                "columns": fields.List(fields.String, required=True)
        }), allow_none=True)
    }, allow_none=True)
}


class ModelTrainingResource(BaseResource):
    pass
model_training_parser_args = {
    "project_id": fields.Str(required=True),
    "model_training_id": fields.Str(required=True),
    "label_url_list": fields.List(fields.Nested({
        "product_id": fields.Str(allow_none=True),
        "label_result_url": fields.Str(required=True),
    }, required=True), required=True),
    "feature_result_url": fields.Str(required=True),
    "business_desc": fields.Nested({
        "problem_type": fields.Str(required=True),
        "target_definition": fields.Nested({
            "product_purchase_para": fields.Nested({
                "product_column": fields.Str(required=True),
                "product_list": fields.List(fields.String, required=True)
            }, allow_none=True),
        }, allow_none=True),
    }, required=True),
    "model_desc": fields.Nested({
        "test_size": fields.Integer(allow_none=True),
        "model_para": fields.Nested({
            "depth": fields.Integer(allow_none=True),
            "learning_rate": fields.Float(allow_none=True),
            "reg_lambda": fields.Float(allow_none=True),
            "iterations": fields.Integer(allow_none=True),
            "one_hot_max_size": fields.Integer(allow_none=True),
            "model_loss_function": fields.Str(allow_none=True),
            "model_eval_metric": fields.Str(allow_none=True),
            "rsm": fields.Float(allow_none=True),
        }, allow_none=True)
    }, allow_none=True)
}


class ModelDeployingResource(BaseResource):
    pass
model_deploying_parser_args = {
    "model_url": fields.Str(required=True)
}


class ModelPredictionResource(BaseResource):
    pass
model_prediction_parser_args = {
    "model_url": fields.Str(required=True),
    "prediction_id": fields.Str(required=True),
    "entity_set_desc": entity_set_preview_parser_args['entity_set_desc']
}