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
    WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback,
    TrainingArguments, TrainerState, TrainerControl)
from peft import prepare_model_for_int8_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'

    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="openai/whisper-large-v2")
    parser.add_argument("--language", type=str, default="Marathi")
    parser.add_argument("--learning_rate", type=str, default=1e-4)
    parser.add_argument("--weight_decay", type=str, default=0.005)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--is_8_bit", type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default="/tmp/data")

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

    task = "transcribe"
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=task)
    metric = evaluate.load("wer")


    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=task)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    if args.is_8_bit:
        model = WhisperForConditionalGeneration.from_pretrained(args.model_name, load_in_8bit=True)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model_name)


    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if args.is_8_bit:
        model = prepare_model_for_int8_training(model)

    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,  # change to a repo name of your choice
        per_device_train_batch_size=int(args.train_batch_size),
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=float(args.learning_rate),
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        fp16=True,
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=128,
        logging_steps=25,
        remove_unused_columns=False,
        # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
    )

    print(training_args)
    
    class SavePeftCheckpointCallback(TrainerCallback):
        def on_save(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control


    class SavePeftTrainEndCallback(TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            peft_model_path = os.path.join(state.best_model_checkpoint, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(state.best_model_checkpoint, "pytorch_model.bin")
            os.remove(pytorch_model_path) if os.path.exists(pytorch_model_path) else None
            return control

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset.get("test", train_dataset["test"]),
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        # callbacks=[SavePeftTrainEndCallback],
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    logger.info("Start training")
    transformers.logging.set_verbosity_error()
    
    logger.info("Data loaded")

    trainer.train()

    logger.info("Start evaluating")
    eval_result = trainer.evaluate(eval_dataset=train_dataset["test"])

    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    logger.info("Start saving")
    # Saves the model to s3
    # trainer.save_model(args.model_dir)
    # processor.tokenizer.save_pretrained(args.model_dir)
    
    peft_model_path = os.path.join(args.model_dir, "adapter_model")
    os.makedirs(peft_model_path, exist_ok=True)
    model.save_pretrained(peft_model_path)
    pytorch_model_path = os.path.join(peft_model_path, "pytorch_model.bin")
    if os.path.exists(pytorch_model_path):
        os.remove(pytorch_model_path)

    logger.info("All done")