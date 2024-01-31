'''
Reference: https://huggingface.co/datasets/pierresi/cord/blob/main/cord.py
'''
import os
import datasets
logger = datasets.logging.get_logger(__name__)

from data.cord.graph_cord import graph_builder


def get_dataset_folder(ds_name):
    path = os.getcwd()
    if "GraphLayoutLM" in path:
        while True:
            path, folder = os.path.split(path)
            if folder == "GraphLayoutLM":
                break
    dest = os.path.join(path, "GraphLayoutLM", "datasets", ds_name)
    return dest


class BaseDataset(datasets.GeneratorBasedBuilder):
    ds_name = "base"
    tags_names = []
    label_2_id = {tags_names[i]: i for i in range(len(tags_names))}

    citation = ""
    desctiption = ""
    homepage = ""

    def _info(self):
        return datasets.DatasetInfo(
            description=self.desctiption,
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
                    "ner_tags": datasets.Sequence(datasets.features.ClassLabel(names=self.tags_names)),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            citation=self.citation,
            homepage=self.homepage,
        )

    @staticmethod
    def get_line_bbox(bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]
        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def get_dest(self):
        return get_dataset_folder(self.ds_name)

    @staticmethod
    def build_graph(dest):
        if not os.path.exists(os.path.join(dest, "train", "graph")):
            print(f"Building graph in {dest}...")
            graph_builder(dest)

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators. Uses local files located with data_dir"""
        dest = self.get_dest()
        print(f'DEST: {dest}')
        self.build_graph(dest)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": dest + "/train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dest + "/dev"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": dest + "/test"}
            ),
        ]

    @staticmethod
    def process_file(file, graph_dir, ann_dir, img_dir):
        raise NotImplemented

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "reordered_json")
        graph_dir = os.path.join(filepath, "graph")
        img_dir = os.path.join(filepath, "image")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            record = self.process_file(file, graph_dir, ann_dir, img_dir)
            if record is None:
                continue
            record["id"] = str(guid)
            yield guid, record
