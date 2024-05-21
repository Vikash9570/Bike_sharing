from src.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.logger import logging
# from src.pipeline 





STAGE_NAME="Data Ingestion Stage"
try:
    logging.info(f">> stage {STAGE_NAME} started")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logging.info(f'>> stage {STAGE_NAME} completed')
except Exception as e:
    logging.exception(e)
    raise e