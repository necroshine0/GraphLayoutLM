import os
import json
import datasets
from data.cord.base_dataset import BaseDataset
from utils.image_utils import load_image, normalize_bbox, quad_to_box


class SberConfig(datasets.BuilderConfig):
    """BuilderConfig for SBER"""
    def __init__(self, **kwargs):
        """BuilderConfig for SBER.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SberConfig, self).__init__(**kwargs)


class SberDataset(BaseDataset):
    BUILDER_CONFIGS = [
        SberConfig(name="sber-slides", version=datasets.Version("1.0.0"), description="SBER dataset"),
    ]

    ds_name = "sber-slides"
    tags_names = [
        "type: focus",
        "type: label",
        "type: list, flavour: bul_list",
        "type: list, flavour: enum_list",
        "type: pic",
        "type: pic, flavour: icon",
        "type: plot",
        "type: subtitle",
        "type: table, flavour: mesh",
        "type: table, flavour: mesh, subelement: cell",
        "type: table, flavour: regular_table",
        "type: text",
        "type: timeline",
        "type: title",
    ]

    @staticmethod
    def process_file(file, graph_dir, ann_dir, img_dir):
        words, bboxes, ner_tags, node_ids = [], [], [], []
        with open(os.path.join(graph_dir, file), "r", encoding="utf8") as f:
            graph_data = json.load(f)
        edges = graph_data["edges"]
        if len(edges) == 0:
            print("\nlen error:", os.path.join(graph_dir, file))
            # exit(0)
            return None

        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        file_name = os.path.join(img_dir, file)
        file_name = file_name.replace("json", "png")
        _, size = load_image(file_name)
        for i, item in enumerate(data["valid_line"]):
            cur_line_bboxes = []
            line_words, label = item["words"], item["category"]
            line_words = [w for w in line_words if w["text"].strip() != ""]
            if len(line_words) == 0:
                continue

            for w in line_words:
                words.append(w["text"])
                ner_tags.append(label)
                cur_line_bboxes.append(normalize_bbox(quad_to_box(w["quad"]), size))
                node_ids.append(item["id"])

            # by default: --segment_level_layout 1
            # if do not want to use segment_level_layout, comment the following line
            cur_line_bboxes = SberDataset.get_line_bbox(cur_line_bboxes)
            bboxes.extend(cur_line_bboxes)

        return {
            "words": words,
            "bboxes": bboxes,
            "ner_tags": ner_tags,
            "node_ids": node_ids,
            "edges": edges,
            "file_name": file_name
        }
