from src.text_summarizer.logging import logger
from src.text_summarizer.pipeline.stage_1_data_ingestion import (
    DataIngestionTrainingPipeline,
)

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>> stage {STAGE_NAME} Completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
