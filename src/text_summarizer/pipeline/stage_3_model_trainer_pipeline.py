from src.text_summarizer.components.model_trainer import ModelTrainer
from src.text_summarizer.config.configuration import ConfigurationManager
from src.text_summarizer.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(self) -> None:
        pass

    def initiate_model_trainer(self) -> None:
        config_params = ConfigurationManager()
        model_trainer_config = config_params.get_model_trainer_config()
        model_trainer_params = config_params.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config, params=model_trainer_params)
        model_trainer.train()
