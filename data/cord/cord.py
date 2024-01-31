'''
Reference: https://huggingface.co/datasets/pierresi/cord/blob/main/cord.py
'''

import os
import json
import datasets
from data.cord.base_dataset import BaseDataset
from utils.image_utils import load_image, normalize_bbox, quad_to_box

logger = datasets.logging.get_logger(__name__)



CORD_CITATION = """\
@article{park2019cord,
  title={CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
"""


class CordConfig(datasets.BuilderConfig):
    """BuilderConfig for CORD"""
    def __init__(self, **kwargs):
        """BuilderConfig for CORD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CordConfig, self).__init__(**kwargs)


class CordDataset(BaseDataset):
    BUILDER_CONFIGS = [
        CordConfig(name="cord", version=datasets.Version("1.0.0"), description="CORD dataset"),
    ]

    ds_name = "CORD"
    tags_names = ["O", "B-MENU.NM", "B-MENU.NUM", "B-MENU.UNITPRICE", "B-MENU.CNT", "B-MENU.DISCOUNTPRICE", "B-MENU.PRICE",
             "B-MENU.ITEMSUBTOTAL", "B-MENU.VATYN", "B-MENU.ETC", "B-MENU.SUB_NM", "B-MENU.SUB_UNITPRICE",
             "B-MENU.SUB_CNT", "B-MENU.SUB_PRICE", "B-MENU.SUB_ETC", "B-VOID_MENU.NM", "B-VOID_MENU.PRICE",
             "B-SUB_TOTAL.SUBTOTAL_PRICE", "B-SUB_TOTAL.DISCOUNT_PRICE", "B-SUB_TOTAL.SERVICE_PRICE",
             "B-SUB_TOTAL.OTHERSVC_PRICE", "B-SUB_TOTAL.TAX_PRICE", "B-SUB_TOTAL.ETC", "B-TOTAL.TOTAL_PRICE",
             "B-TOTAL.TOTAL_ETC", "B-TOTAL.CASHPRICE", "B-TOTAL.CHANGEPRICE", "B-TOTAL.CREDITCARDPRICE",
             "B-TOTAL.EMONEYPRICE", "B-TOTAL.MENUTYPE_CNT", "B-TOTAL.MENUQTY_CNT", "I-MENU.NM", "I-MENU.NUM",
             "I-MENU.UNITPRICE", "I-MENU.CNT", "I-MENU.DISCOUNTPRICE", "I-MENU.PRICE", "I-MENU.ITEMSUBTOTAL",
             "I-MENU.VATYN", "I-MENU.ETC", "I-MENU.SUB_NM", "I-MENU.SUB_UNITPRICE", "I-MENU.SUB_CNT",
             "I-MENU.SUB_PRICE", "I-MENU.SUB_ETC", "I-VOID_MENU.NM", "I-VOID_MENU.PRICE", "I-SUB_TOTAL.SUBTOTAL_PRICE",
             "I-SUB_TOTAL.DISCOUNT_PRICE", "I-SUB_TOTAL.SERVICE_PRICE", "I-SUB_TOTAL.OTHERSVC_PRICE",
             "I-SUB_TOTAL.TAX_PRICE", "I-SUB_TOTAL.ETC", "I-TOTAL.TOTAL_PRICE", "I-TOTAL.TOTAL_ETC",
             "I-TOTAL.CASHPRICE", "I-TOTAL.CHANGEPRICE", "I-TOTAL.CREDITCARDPRICE", "I-TOTAL.EMONEYPRICE",
             "I-TOTAL.MENUTYPE_CNT", "I-TOTAL.MENUQTY_CNT"]

    citation = CORD_CITATION
    description = "https://github.com/clovaai/cord/"
    homepage = "https://github.com/clovaai/cord/"

    @staticmethod
    def process_file(file, graph_dir, ann_dir, img_dir):
        words, bboxes, ner_tags, node_ids, edges = [], [], [], [], []
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
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)
        for item in data["valid_line"]:
            cur_line_bboxes = []
            line_words, label = item["words"], item["category"]
            line_words = [w for w in line_words if w["text"].strip() != ""]
            if len(line_words) == 0:
                continue
            if label == "other":
                for w in line_words:
                    words.append(w["text"])
                    ner_tags.append("O")
                    cur_line_bboxes.append(normalize_bbox(quad_to_box(w["quad"]), size))
                    node_ids.append(item["id"])
            else:
                words.append(line_words[0]["text"])
                ner_tags.append("B-" + label.upper())
                cur_line_bboxes.append(normalize_bbox(quad_to_box(line_words[0]["quad"]), size))
                node_ids.append(item["id"])
                for w in line_words[1:]:
                    words.append(w["text"])
                    ner_tags.append("I-" + label.upper())
                    cur_line_bboxes.append(normalize_bbox(quad_to_box(w["quad"]), size))
                    node_ids.append(item["id"])
            # by default: --segment_level_layout 1
            # if do not want to use segment_level_layout, comment the following line
            cur_line_bboxes = CordDataset.get_line_bbox(cur_line_bboxes)
            bboxes.extend(cur_line_bboxes)

        return {
            "words": words,
            "bboxes": bboxes,
            "ner_tags": ner_tags,
            "node_ids": node_ids,
            "edges": edges,
            "image": image,
            "image_path": image_path
        }
