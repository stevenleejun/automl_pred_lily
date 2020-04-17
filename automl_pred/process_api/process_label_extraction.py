from ..process.label_extraction_process import LabelExtractionProcess as ProcessClass


def process_label_extraction(process_config, logger=None, root_config_path='../config/config_automl_pred_mlplat/root_config.yml'):
    process_model = ProcessClass(
        process_config=process_config,
        root_config_path=root_config_path,
        logger=logger
    )
    result = process_model.process()
    return result
