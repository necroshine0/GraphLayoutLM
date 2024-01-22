import os
import json
import datasets
from data.cord.graph_cord import graph_builder
from utils.image_utils import load_image, normalize_bbox, quad_to_box

logger = datasets.logging.get_logger(__name__)


class SberConfig(datasets.BuilderConfig):
    """BuilderConfig for SBER"""
    def __init__(self, **kwargs):
        """BuilderConfig for SBER.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SberConfig, self).__init__(**kwargs)


class Sber(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SberConfig(name="sber-slides", version=datasets.Version("1.0.0"), description="SBER dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id":       datasets.Value("string"),
                    "words":    datasets.Sequence(datasets.Value("string")),
                    "bboxes":   datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "node_ids": datasets.Sequence(datasets.Value("int64")),
                    "edges":    datasets.Sequence(
                        {   
                            "head": datasets.Value("int64"),
                            "tail": datasets.Value("int64"),
                            "rel":  datasets.Value("string"),
                        }
                    ),
                    "ner_tags":  datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
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
                        )
                    ),
                    "image":      datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators. Uses local files located with data_dir"""
        dest = r"../datasets/sber-slides"
        if not os.path.exists(os.path.join(dest, "train", "graph")):
            graph_builder(dest)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": dest + "/train"}
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dest + "/dev"}
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": dest + "/test"}
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]
        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "reordered_json")
        graph_dir = os.path.join(filepath, "graph")
        img_dir = os.path.join(filepath, "image")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            words, bboxes, ner_tags, node_ids = [], [], [], []
            with open(os.path.join(graph_dir, file), "r", encoding="utf8") as f:
                graph_data = json.load(f)
            edges = graph_data["edges"]
            if len(edges)==0:
                print("\nlen error:", os.path.join(graph_dir, file))
                # exit(0)
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
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                bboxes.extend(cur_line_bboxes)
            # yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}
            yield guid, {"id": str(guid),
                         "words": words,
                         "bboxes": bboxes,
                         "ner_tags": ner_tags,
                         "node_ids": node_ids,
                         "edges": edges,
                         "image": image,
                         "image_path": image_path
                    }
