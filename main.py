from src.text_summarizer.logging import logger
from src.text_summarizer.pipeline.stage_1_data_ingestion_pipeline import (
    DataIngestionTrainingPipeline,
)
from src.text_summarizer.pipeline.stage_2_data_transformation_pipeline import (
    DataTransformationTrainingPipeline,
)

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f">>>>>> stage {STAGE_NAME} Completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
    data_transformation_pipeline = DataTransformationTrainingPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f">>>>>> stage {STAGE_NAME} Completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
