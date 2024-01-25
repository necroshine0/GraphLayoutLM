import os
from data.cord.sber import SberDataset
from data.cord.base_dataset import get_dataset_folder
from detectron2.data import MetadataCatalog, DatasetCatalog


def get_sber_dict(dataset_folder, set):
    # Build graph if is not built
    SberDataset.build_graph(dataset_folder)

    # Get dataset_dicts
    dataset_dicts = []
    set_dir = os.path.join(dataset_folder, set)
    ann_dir = os.path.join(set_dir, "reordered_json")
    graph_dir = os.path.join(set_dir, "graph")
    img_dir = os.path.join(set_dir, "image")
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        record = SberDataset.process_file(file, graph_dir, ann_dir, img_dir)
        if record is None:
            continue
        record["id"] = str(guid)
        dataset_dicts.append(record)
    return dataset_dicts


def main():
    thing_classes = SberDataset.tags_names
    dataset_folder = get_dataset_folder("sber-slides")
    print('Dataset folder:', dataset_folder)

    for set_type in ["train", "val"]:
        print(f'Building {set_type} set...')
        folder_name = "sberslides_" + set_type

        DatasetCatalog.register(folder_name, lambda set_type: get_sber_dict(dataset_folder, set_type))
        MetadataCatalog.get(folder_name).set(thing_classes=thing_classes)
        print("Done!\n")

    sber_metadata = MetadataCatalog.get("sberslides_train")
    dataset_dicts = get_sber_dict(dataset_folder, "train")
    print(sber_metadata)
    print()
    print(dataset_dicts[0])


if __name__ == "__main__":
    main()
