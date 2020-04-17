from ..process.model_prediction_process import ModelPredictionProcess as ProcessClass


def process_model_prediction(process_config, logger=None, root_config_path='../config/config_automl_pred_mlplat/root_config.yml'):
    process_model = ProcessClass(
        process_config=process_config,
        root_config_path=root_config_path,
        logger=logger
    )
    result = process_model.process()
    return result
