import PIL.Image
import transformers
from datasets import load_dataset, DatasetDict, load_metric
from datasets.utils.file_utils import get_datasets_user_agent
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import PIL
from typing import Optional, Dict, Any, List
import torch
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer

USER_AGENT = get_datasets_user_agent()
repo_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(repo_id)

def fetch_single_image(image_url: str, timeout: bool = None, retries: int = 0) -> Optional[PIL.Image.Image]:
    """Fetch an image from url on internet.

    Args:
        image_url (str): URL to image.
        timeout (bool, optional): Timeout for requests. Defaults to None.
        retries (int, optional): Num. of retries if fails. Defaults to 0.

    Returns:
        Optional[PIL.Image.Image]: Downloaded Image if exists.
    """
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image

def fetch_images(batch: dict, num_threads: int, timeout: int = None, retries: int = 0) -> Dict[str, Any]:
    """Fetch images from a Dataset's batch 

    Args:
        batch (dict): Batch of data.
        num_threads (int): Num of threads for multithreading.
        timeout (int, optional): Timeout for requests. Defaults to None.
        retries (int, optional): Num. of retries if fails. Defaults to 0.

    Returns:
        Dict[str, Any]: Batch data with Images.
    """
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["imageurl"]))
    return batch

def transform(example_batch: dict) -> Dict[str, Any]:
    """Transform a batch computing pixel values and ViTFeatureExtractor transformations.

    Args:
        example_batch (dict): Batch of data.

    Returns:
        Dict[str, Any]: Batch of data with pixel values.
    """
    # Feature extractor 
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors = 'pt') #pt es tipo pytorch
    inputs["labels"] = example_batch["labels"]
    return inputs

def generate_datasets(dataset_name: str, n_samples: int) -> DatasetDict:
    """Generate datasets for train, test and validation.

    Args:
        dataset_name (str): Dataset name in HuggingFace.
        n_samples (int): Number of samples to generate.

    Returns:
        DatasetDict: Train, test and validation split.
    """
    num_threads = 20
    dset = load_dataset(dataset_name)
    dset = dset.shuffle(42)["train"].select(range(n_samples))
    dset = dset.map(fetch_images, batched=True, batch_size=100, fn_kwargs={"num_threads": num_threads})

    dset = dset.remove_columns(column_names=['website_name', 'competence_date', 'country_code', 'currency_code', 'brand', 'category1_code', 'category2_code', 'product_code', 'title', 'itemurl', 'full_price', 'price', 'full_price_eur', 'price_eur', 'flg_discount'])
    dset = dset.rename_column(new_column_name="labels", original_column_name="category3_code")
    dset = dset.class_encode_column("labels")

    # Train test split
    dset = dset.train_test_split(test_size=0.3)
    ds_train = dset["train"]
    test = dset["test"].train_test_split(test_size=0.3)
    ds_test = test["train"]
    ds_val = test["test"]

    train_test_valid_dataset = DatasetDict(
        {
            'train': ds_train,
            'test': ds_test,
            'valid': ds_val
        }
    )
    return train_test_valid_dataset

def collate_fn(batch: List[dict]) -> Dict[str, torch.tensor]:
    """Collate and return pixel values from batch dicts as Tensors.

    Args:
        batch (List[dict]): Batch to process

    Returns:
        Dict[str, torch.tensor]: Processed batch with pixel values and labels
    """  

    return {
        "pixel_values": torch.stack([torch.tensor(x["pixel_values"]) for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }

def compute_metrics(prediction):
    """Compute classification metrics."""
    metric = load_metric("accuracy")
    return metric.compute(predictions = np.argmax(prediction.predictions, axis =1), references = prediction.label_ids)

def train_model() -> None:
    """Train model."""
    # Datasets and labels
    train_test_valid_dataset = generate_datasets("DBQ/Matches.Fashion.Product.prices.France", 10000)
    labels = train_test_valid_dataset["train"].features["labels"].names

    # Model
    model = ViTForImageClassification.from_pretrained(
        repo_id,
        num_labels = len(labels),
        id2label = {int(i): c for i, c in enumerate(labels)},
        label2id = {c: int(i) for i, c in enumerate(labels)})

    # Training arguments
    training_args = TrainingArguments(
        output_dir = './vit-clothes-classification', # output en HuggingFace
        evaluation_strategy='steps',
        num_train_epochs = 8,
        learning_rate = 2e-4,
        remove_unused_columns = False,
        push_to_hub = True,
        load_best_model_at_end = True
    )

    # Trainer object
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator= collate_fn,
        compute_metrics = compute_metrics,
        train_dataset = train_test_valid_dataset["train"],
        eval_dataset = train_test_valid_dataset["valid"],
        tokenizer = feature_extractor
    )

    # Train model, save it and log metrics
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    # Log eval metricas
    metrics = trainer.evaluate(train_test_valid_dataset["test"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Push model to HF hub
    kwargs = {
        "finetuned_from": model.config._name_or_path,
        "tasks": "image-classification",
        "dataset": 'DBQ/Matches.Fashion.Product.prices.France',
        "tags":["image-classification", "clothes-classification"]
    }
    trainer.push_to_hub(commit_message = "VIT model tuned", **kwargs)

if __name__ == "__main__":
    train_model()