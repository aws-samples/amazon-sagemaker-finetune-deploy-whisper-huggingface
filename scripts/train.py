import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import load_from_disk
import transformers
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--learning_rate", type=str, default=1e-4)
    parser.add_argument("--weight_decay", type=str, default=0.005)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args, _ = parser.parse_known_args()

    # set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    print(args.training_dir)
    train_dataset = load_from_disk(args.training_dir)

    logger.info("Data loaded")

    # add tokenizer, feature extractor and processor
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        tokenizer = processor.tokenizer

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,  # change to a repo name of your choice
        per_device_train_batch_size=int(args.train_batch_size),
        gradient_accumulation_steps=16//args.train_batch_size,  # increase by 2x for every 2x decrease in batch size
        learning_rate=float(args.learning_rate),
        warmup_steps=args.warmup_steps,
        max_steps=int(args.max_steps),
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        logging_dir=f"{args.output_data_dir}/logs",
        dataloader_num_workers=args.dataloader_num_workers,
        save_total_limit=4,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset.get("validation", train_dataset["test"]),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    logger.info("Start training")
    transformers.logging.set_verbosity_error()

    trainer.train()

    logger.info("Start evaluating")
    eval_result = trainer.evaluate(eval_dataset=train_dataset["test"])

    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    logger.info("Start saving")
    # Saves the model to s3
    trainer.save_model(args.model_dir)
    processor.tokenizer.save_pretrained(args.model_dir)

    logger.info("All done")
