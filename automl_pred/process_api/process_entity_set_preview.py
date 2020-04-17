from ..process.entity_set_preview_process import EntitySetPreviewProcess as ProcessClass


def process_entity_set_preview(process_config, logger=None, root_config_path='../config/config_automl_pred_mlplat/root_config.yml'):
    process_model = ProcessClass(
        process_config=process_config,
        root_config_path=root_config_path,
        logger=logger
    )
    result = process_model.process()
    return result
