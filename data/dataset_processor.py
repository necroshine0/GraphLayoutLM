import os
import torch
import numpy as np
from torchvision import transforms

from utils.graph_utils import set_nodes, set_edges
from utils.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from datasets.arrow_dataset import Dataset
from datasets import ClassLabel, Features, Sequence, load_dataset

from detectron2.structures.boxes import BoxMode


class DatasetProcessor(object):
    def __init__(self, args_namespace, boxmode='xyxy'):
        self.args = args_namespace
        # see dataset mapper, better not to use processing in dataset_processor
        self.boxmode = BoxMode(0) if boxmode == 'xyxy' else BoxMode(1)  # xyxy or xywh

        self.column_names = None
        self.text_column_name = None
        self.label_column_name = None
        self.label_list = None
        self.label_to_id = None
        self.tokenizer = None

    def init_meta(self, thing_classes):
        self.column_names = ['id', 'words', 'bboxes', 'node_ids', 'edges', 'ner_tags', 'image', 'image_path']
        self.label_column_name = 'ner_tags'
        self.label_list = thing_classes
        self.text_column_name = "words" if "words" in self.column_names else "tokens"

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_and_align_labels(self, examples, augmentation=False):
        tokenized_inputs = self.tokenizer(
            examples[self.text_column_name], boxes=examples["bboxes"], word_labels=examples["ner_tags"],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
        )

        if self.args.visual_embed:
            common_transform, patch_transform = self.get_transforms()

        labels, bboxes, images, image_paths, nodes, edges = [], [], [], [], [], []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[self.label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            node_ids = examples["node_ids"][org_batch_index]
            ipath = examples["image_path"][org_batch_index]

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
                    label_ids.append(label[word_idx])
                    bbox_inputs.append(bbox[word_idx])
                    new_node_ids.append(node_ids[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if self.args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                    new_node_ids.append(node_ids[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            image_paths.append(ipath)
            if self.args.visual_embed:
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)

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

        if self.args.annotation_tag:
            annotations = []
            assert len(bboxes) == len(labels)
            for i in range(len(bboxes)):
                assert len(bboxes[i]) == len(labels[i])
                annotations.append(
                    [{"bbox": bboxes[i][j], "category_id": labels[i][j], "bbox_mode": self.boxmode}
                     for j in range(len(bboxes[i]))]
                )

            tokenized_inputs["annotations"] = annotations
        else:
            tokenized_inputs["labels"] = labels
            tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["image_path"] = image_paths
        if self.args.visual_embed:
            tokenized_inputs["images"] = images
        return tokenized_inputs

    def get_transforms(self):
        imagenet_default_mean_and_std = self.args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        common_transform = Compose([
            # transforms.ColorJitter(0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=self.args.input_size, interpolation=self.args.train_interpolation),
        ])

        patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
        return common_transform, patch_transform

    def process_dataset(self, dataset):
        # Tokenize all texts and align the labels with them.
        if hasattr(dataset, 'keys'):
            for split in list(dataset.keys()):
                dataset[split] = dataset[split].map(
                    self.tokenize_and_align_labels,
                    batched=True,
                    remove_columns=self.column_names,
                    num_proc=self.args.preprocessing_num_workers,
                    load_from_cache_file=not self.args.overwrite_cache,
                )
        else:
            dataset = dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                remove_columns=self.column_names,
                num_proc=self.args.preprocessing_num_workers,
                load_from_cache_file=not self.args.overwrite_cache,
            )
        return dataset


def get_dataset_dict(dataset_folder, split, tokenizer, args):
    # dataset_folder = ".../datasets/dataset_name"
    dataset_name = os.path.split(dataset_folder)[-1]
    if dataset_name == "CORD":
        from data.cord.sber import CordDataset as UsedDataset
    elif dataset_name == "sber-slides":
        from data.cord.sber import SberDataset as UsedDataset
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")
    # Build graph if is not built
    UsedDataset.build_graph(dataset_folder)

    # Get dataset_dicts
    dataset_list = []
    set_dir = os.path.join(dataset_folder, split)
    img_dir = os.path.join(set_dir, "image")
    graph_dir = os.path.join(set_dir, "graph")
    ann_dir = os.path.join(set_dir, "reordered_json")
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        record = UsedDataset.process_file(file, graph_dir, ann_dir, img_dir)
        if record is None:
            continue
        record["id"] = str(guid)
        dataset_list.append(record)

    thing_classes = UsedDataset.tags_names
    label_to_id = {thing_classes[i]: i for i in range(len(thing_classes))}
    def mapping(elem):
        elem['ner_tags'] = list(map(lambda x: label_to_id[x], elem['ner_tags']))
        # list of dicts to dict of lists
        elem['edges'] = {k: [dic[k] for dic in elem['edges']] for k in ['head', 'tail', 'rel']}
        return elem
    dataset_list = list(map(mapping, dataset_list))

    dataset = Dataset.from_list(dataset_list)
    dataset = process_dataset(dataset, thing_classes, tokenizer, args)
    return dataset.to_list()


def process_dataset(dataset, thing_classes, tokenizer, args):
    builder = DatasetProcessor(args)
    builder.init_meta(thing_classes)
    builder.init_tokenizer(tokenizer)
    return builder.process_dataset(dataset)


def load_dataset_from_name(dataset_name):
    if dataset_name == 'cord':
        import data.cord.cord
        from data.cord.cord import CordDataset
        thing_classes = CordDataset.tags_names
        datasets = load_dataset(os.path.abspath(data.cord.cord.__file__), trust_remote_code=True)
    elif 'sber' in dataset_name:
        import data.cord.sber
        from data.cord.sber import SberDataset
        thing_classes = SberDataset.tags_names
        datasets = load_dataset(os.path.abspath(data.cord.sber.__file__), trust_remote_code=True)
    else:
        raise NotImplementedError()
    return datasets, thing_classes


def main(dataset_name):
    datasets, thing_classes = load_dataset_from_name(dataset_name)

    from model.tokenization_graphlayoutlm_fast import GraphLayoutLMTokenizerFast
    tokenizer = GraphLayoutLMTokenizerFast.from_pretrained(
        "pretrained/graphlayoutlm-base",
        tokenizer_file=None,
        use_fast=True,
        add_prefix_space=True,
        revision="main",
    )

    # datasets = process_datasets(datasets, thing_classes, tokenizer, args, for_detectron=False)
