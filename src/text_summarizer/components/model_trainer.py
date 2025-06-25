from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq

from src.text_summarizer.entity import ModelTrainerConfig
from src.text_summarizer.logging import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, params: ModelTrainerConfig) -> None:
        self.config = config
        self.params = params

    def train(self) -> None:

        if torch.backends.mps.is_available(): # Detect device (MPS for Apple Silicon)
            device = "mps"
        elif torch.cuda.is_available(): # Detect NVIDIA GPU (cuda for GPU)
            device = "cuda"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        # Load preprocessed dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.params.num_train_epochs,
            warmup_steps=self.params.warmup_steps,
            per_device_train_batch_size=self.params.per_device_train_batch_size,
            per_device_eval_batch_size=self.params.per_device_eval_batch_size,
            weight_decay=self.params.weight_decay,
            logging_steps=self.params.logging_steps,
            eval_strategy=self.params.eval_strategy,
            eval_steps=self.params.eval_steps,
            # save_steps=self.params.save_steps,
            gradient_accumulation_steps=self.params.gradient_accumulation_steps,
        )
        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["test"],
            eval_dataset=dataset_samsum_pt["validation"],
        )

        trainer.train()

        ## Save model and tokenizer
        model_pegasus.save_pretrained(Path(self.config.root_dir) / "pegasus-samsum-model")
        tokenizer.save_pretrained(Path(self.config.root_dir) / "tokenizer")
