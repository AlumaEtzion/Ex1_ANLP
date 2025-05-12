import torch
import numpy as np
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
import evaluate
from transformers import Trainer, TrainingArguments, BertForSequenceClassification, EvalPrediction, AutoModelForSequenceClassification
import wandb
from evaluate import load
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser



@dataclass
class CustomArguments:
    max_train_samples: int = -1
    max_eval_samples: int = -1
    max_predict_samples: int = -1
    num_train_epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 16
    do_train: bool = False
    do_predict: bool = False
    model_path: Optional[str] = "bert-base-uncased"



def build_training_args(args: CustomArguments) -> TrainingArguments:
    # run_type = "train" if args.do_train else "predict"
    run_name = f"lr_{args.lr}_epochs_{args.num_train_epochs}_bs_{args.batch_size}" if args.do_train else args.model_path.split("/")[2]
    return TrainingArguments(
        output_dir=f"./anlp_ex1_results/results/{run_name}",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        report_to="wandb",
        run_name=run_name,
        logging_dir="./anlp_ex1_results/logs",
        save_total_limit=1,
    )


def load_tadaet_and_tokenizer():
    dataset = load_dataset("nyu-mll/glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # config = AutoConfig.from_pretrained("bert-base-uncased")

    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2,  config=config)
    return tokenizer, dataset

def tokenize_train(examples):
    return tokenizer(examples['sentence1'],examples['sentence2'],max_length=512,truncation=True)

def tokenize_predict(examples):
    return tokenizer(examples['sentence1'],examples['sentence2'],max_length=512,truncation=True,padding=False)


def split_dataset(dataset):
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']
    return train_dataset, validation_dataset, test_dataset

def compute_metrics(p: EvalPrediction):
    metric = load("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return {"accuracy": result}

def predict(model, test_dataset, trainer, file):
    model.eval()
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(-1)
    sentence1_list = test_dataset['sentence1']
    sentence2_list = test_dataset['sentence2']
    with open(file, "w") as f:
        for s1, s2, label in zip(sentence1_list, sentence2_list, predicted_labels):
            f.write(f"{s1}###{s2}###{label}\n")


def main():
    parser = HfArgumentParser(CustomArguments)
    custom_args = parser.parse_args_into_dataclasses()[0]
    device = torch.device("cuda")

    dataset = load_dataset("nyu-mll/glue", "mrpc")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if custom_args.do_train:
        config = AutoConfig.from_pretrained(custom_args.model_path)
        model = BertForSequenceClassification.from_pretrained(custom_args.model_path, config=config)
        dataset_tokenized = dataset.map(tokenize_train, batched=True, batch_size=custom_args.batch_size)


    elif custom_args.do_predict:
        if not custom_args.model_path:
            raise ValueError("You must specify --model_path for prediction.")
        model = AutoModelForSequenceClassification.from_pretrained(custom_args.model_path)
        dataset_tokenized = dataset.map(tokenize_predict, batched=True, batch_size=custom_args.batch_size)
    model.to(device)

    train_dataset, eval_dataset, test_dataset = split_dataset(dataset_tokenized)

    # Subsample if needed
    if custom_args.max_train_samples != -1:
        train_dataset = train_dataset.select(range(custom_args.max_train_samples))
    if custom_args.max_eval_samples != -1:
        eval_dataset = eval_dataset.select(range(custom_args.max_eval_samples))
    if custom_args.max_predict_samples != -1:
        test_dataset = test_dataset.select(range(custom_args.max_predict_samples))

    training_args = build_training_args(custom_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)

    if custom_args.do_train:
        trainer.train()


    if custom_args.do_predict:
        predict(model, test_dataset, trainer, f"{training_args.run_name}_predictions.txt")




if __name__ == "__main__":
    main()
