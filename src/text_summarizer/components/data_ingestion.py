import zipfile
from pathlib import Path
from urllib import request

from src.text_summarizer.entity import DataIngestionConfig
from src.text_summarizer.logging import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def downlaod_file(self) -> None:
        if not Path(self.config.local_data_file).exists():
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file,
            )
            logger.info("File is downloaded")
        else:
            logger.info("File already exits")

    def extract_zip_file(self) -> None:
        """
        Extracts the zip file into the data directory.

        Args:
            Zip_file_path: str (path to zip file)
        Function returns None.

        """
        unzip_path = self.config.unzip_dir
        Path(unzip_path).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
