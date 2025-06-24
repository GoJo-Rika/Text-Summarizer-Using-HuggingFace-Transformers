from pathlib import Path
from typing import List

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from src.text_summarizer.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns.

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type

    """
    try:
        with Path(path_to_yaml).open() as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool=True):
    """
    Create list of directories.

    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): whether to log directory creation. Defaults to True.

    """
    for path in path_to_directories:
        Path(path).mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")
