"""
Fine-tune openai/whisper-medium on the Leyu Amharic dataset.
Intended to be run from the Colab notebook, but can also run standalone.
"""

import yaml
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data_prep import build_dataset


wer_metric = evaluate.load("wer")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def train(config_path: str = "configs/training_config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]

    processor = WhisperProcessor.from_pretrained(
        model_cfg["base_model"],
        language=model_cfg["language"],
        task=model_cfg["task"],
    )

    model = WhisperForConditionalGeneration.from_pretrained(model_cfg["base_model"])
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    dataset = build_dataset(config, processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=train_cfg["output_dir"],
        max_steps=train_cfg["max_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        fp16=train_cfg["fp16"],
        evaluation_strategy=train_cfg["evaluation_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        predict_with_generate=True,
        generation_max_length=config["generation"]["generation_max_length"],
        push_to_hub=train_cfg["push_to_hub"],
        hub_model_id=train_cfg.get("hub_model_id"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.push_to_hub()
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
