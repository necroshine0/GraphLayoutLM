import os
import torch
from torchvision import transforms

from utils.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import numpy as np
from utils.graph_utils import set_nodes, set_edges

from datasets import ClassLabel, load_dataset


def get_transforms(data_args):
    imagenet_default_mean_and_std = data_args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    common_transform = Compose([
        # transforms.ColorJitter(0.4, 0.4, 0.4),
        # transforms.RandomHorizontalFlip(p=0.5),
        RandomResizedCropAndInterpolationWithTwoPic(
            size=data_args.input_size, interpolation=data_args.train_interpolation),
    ])

    patch_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ])
    return common_transform, patch_transform


def build_datasets(tokenizer, data_args, model_args, training_args):
    if data_args.dataset_name == 'cord':
        import data.cord.cord
        print("PATH:", os.path.abspath(data.cord.cord.__file__))
        datasets = load_dataset(os.path.abspath(data.cord.cord.__file__), trust_remote_code=True, cache_dir=model_args.cache_dir)
    elif 'sber' in data_args.dataset_name:
        import data.cord.sber
        print("PATH:", os.path.abspath(data.cord.sber.__file__))
        datasets = load_dataset(os.path.abspath(data.cord.sber.__file__), trust_remote_code=True, cache_dir=model_args.cache_dir)
    else:
        raise NotImplementedError()

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features

    remove_columns = column_names
    text_column_name = "words" if "words" in column_names else "tokens"
    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}


    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples, augmentation=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name], boxes=examples["bboxes"], word_labels=examples["ner_tags"],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
        )

        if data_args.visual_embed:
            common_transform, patch_transform = get_transforms(data_args)

        labels, bboxes, images, image_paths, nodes, edges = [], [], [], [], [], []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]

            node_ids = examples["node_ids"][org_batch_index]

            previous_word_idx = None
            label_ids, bbox_inputs, new_node_ids = [], [], []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                    new_node_ids.append(-1)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                    new_node_ids.append(node_ids[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                    new_node_ids.append(node_ids[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)

            if data_args.visual_embed:
                ipath = examples["image_path"][org_batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)
                image_paths.append(ipath)

            new_node_data, new_ids = set_nodes(new_node_ids)

            new_edges_data = set_edges(examples["edges"][org_batch_index], new_ids)
            nodes.append(new_node_data)
            edges.append(new_edges_data)

        # build graph mask
        graph_mask_list = []
        input_len = 709
        for nodes_data, edges_data in zip(nodes, edges):
            edges_len = len(edges_data)
            graph_mask = -9e15 * np.ones((input_len, input_len))
            for edge_i in range(edges_len):
                edge = edges_data[edge_i]
                if edge[0] == -1:
                    break
                a_node_index, b_node_index = edge[0], edge[1]
                [a_start, a_end] = nodes_data[a_node_index]
                [b_start, b_end] = nodes_data[b_node_index]
                graph_mask[a_start:a_end + 1, b_start:b_end + 1] = 0
            graph_mask_list.append(graph_mask)
        tokenized_inputs["graph_mask"] = graph_mask_list

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        if data_args.visual_embed:
            tokenized_inputs["images"] = images
            tokenized_inputs["image_path"] = image_paths
        return tokenized_inputs


    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            datasets["train"] = datasets["train"].select(range(data_args.max_train_samples))

    if training_args.do_eval:
        validation_name = "test"
        if validation_name not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_val_samples is not None:
            datasets[validation_name] = datasets[validation_name].select(range(data_args.max_val_samples))

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        if data_args.max_test_samples is not None:
            datasets["test"] = datasets["test"].select(range(data_args.max_test_samples))


    for set in datasets:
        datasets[set] = datasets[set].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    return datasets, label_list
