import os
import json
from detectron2.data import MetadataCatalog, DatasetCatalog
from data.cord.graph_cord import graph_builder
from utils.image_utils import load_image, normalize_bbox, quad_to_box


thing_classes=['type: focus',
               'type: label',
               'type: list, flavour: bul_list',
               'type: list, flavour: enum_list',
               'type: pic',
               'type: pic, flavour: icon',
               'type: plot',
               'type: subtitle',
               'type: table, flavour: mesh',
               'type: table, flavour: mesh, subelement: cell',
               'type: table, flavour: regular_table',
               'type: text',
               'type: timeline',
               'type: title']


def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]
    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox


def get_sber_dict(dataset_folder, set):
    # Build graph if is not built
    if not os.path.exists(os.path.join(dataset_folder, "train", "graph")):
        print("Building graph...")
        graph_builder(dataset_folder)
        print("Done!")

    set_dir = os.path.join(dataset_folder, set)
    ann_dir = os.path.join(set_dir, "reordered_json")
    graph_dir = os.path.join(set_dir, "graph")
    img_dir = os.path.join(set_dir, "image")

    # Get dataset_dicts
    dataset_dicts = []
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        record = {}

        words, bboxes, ner_tags, node_ids = [], [], [], []
        with open(os.path.join(graph_dir, file), "r", encoding="utf8") as f:
            graph_data = json.load(f)
        edges = graph_data["edges"]
        if len(edges)==0:
            print("\nlen error:", os.path.join(graph_dir, file))
            continue

        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)
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
            cur_line_bboxes = get_line_bbox(cur_line_bboxes)
            bboxes.extend(cur_line_bboxes)

        record["id"] = str(guid)
        record["words"] = words
        record["bboxes"] = bboxes
        record["ner_tags"] = ner_tags
        record["node_ids"] = node_ids
        record["edges"] = edges
        record["image"] = image
        record["image_path"] = image_path
        dataset_dicts.append(record)
    return dataset_dicts


def main():
    for set_type in ["train", "val"]:
        print(f'Building {set_type} set...')
        folder_name = "sberslides_" + set_type
        DatasetCatalog.register(folder_name, lambda set_type: get_sber_dict("datasets/sber-slides", set_type))
        MetadataCatalog.get(folder_name).set(thing_classes=thing_classes)
        print("Done!\n")

    sber_metadata = MetadataCatalog.get("sberslides_train")
    dataset_dicts = get_sber_dict("datasets/sber-slides", "train")


if name == "__main__":
    main()