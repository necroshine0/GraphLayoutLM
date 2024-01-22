import os
import re
import json
import shutil
from glob import glob
from sklearn.model_selection import train_test_split


# FIXME: paths
PATH_PREFIX = ""
USE_DEV = False

def main():
    os.makedirs(f"{PATH_PREFIX}sber-slides/json", exist_ok=True)
    os.makedirs(f"{PATH_PREFIX}sber-slides/train/image", exist_ok=True)
    os.makedirs(f"{PATH_PREFIX}sber-slides/train/json", exist_ok=True)
    os.makedirs(f"{PATH_PREFIX}sber-slides/test/image", exist_ok=True)
    os.makedirs(f"{PATH_PREFIX}sber-slides/test/json", exist_ok=True)
    if USE_DEV:
        os.makedirs(f"{PATH_PREFIX}sber-slides/dev/image", exist_ok=True)
        os.makedirs(f"{PATH_PREFIX}sber-slides/dev/json", exist_ok=True)

    with open(f"{PATH_PREFIX}sber-slides/result.json", 'r') as f:
        sber = json.load(f)
    images = sber['images']
    categories = sber['categories']
    annotations = sber['annotations']


    # To number the images
    for i, img in enumerate(images):
        img_file = img['file_name']
        new_file = re.sub(r'/(\w+)-', f'/{i}-', img_file)
        img['file_name'] = new_file
        try:
            os.rename(f"sber-slides/{img_file}", f"sber-slides/{new_file}")
        except:
            continue

    # Build and save annotations in CORD format
    for img in images:
        img_id = img['id']
        img_file = img['file_name']
        W, H = img['width'], img['height']
        # Group annotations by image
        img_annots = []
        for ann in annotations:
            if ann['image_id'] == img_id:
                img_annots.append(ann)

        # Filter objects with less than 2 elements
        if len(img_annots) <= 1:
            continue

        valid_lines = []
        for ann in img_annots:
            category_id = ann['category_id']
            category = categories[ann['category_id']]['name']
            x1, y1, w, h = ann['bbox']
            # Filter invalid bounding boxes
            if x1 + w > W or y1 + h > H:
                continue

            quad = {
                "x1": x1, "y1": y1,
                "x2": x1 + w, "y2": y1,
                "x3": x1 + w, "y3": y1 + h,
                "x4": x1, "y4": y1 + h,
            }

            word = {
                "quad": quad,
                "is_key": 0,
                "row_id": int((2 * y1 + h) / 2),
                "text": "---",  # no text in annotations
            }

            cord_like_ann = {
                "words": [word],
                "category_id": category_id,
                "category": category,
                "group_id": -1,  # FIXME: unknown yet
            }

            valid_lines.append(cord_like_ann)

        img_json = {
            "valid_line": valid_lines,
            "meta": {
                "version": "v1.0",
                "split": "UNK",
                "image_id": img_id,
                "file_name": img_file,
                "image_size": {
                    "width": img['width'],
                    "height": img['height']
                }
            },
        }

        # Saving
        file = os.path.split(img_file)[-1].replace('.png', '.json')
        with open(f"{PATH_PREFIX}sber-slides/json/{file}", 'w') as f:
            json.dump(img_json, f)


    # Split train/test
    files = glob(f"{PATH_PREFIX}sber-slides/json/*")
    print(len(files))

    files_train, files_test = train_test_split(files, test_size=31, random_state=10)
    files_dev = []
    if USE_DEV:
        files_test, files_dev = train_test_split(files, test_size=1, random_state=10)
    sets_dict = {"train": files_train, "dev": files_dev, "test": files_test}

    for set in sets_dict:
        base_files = sets_dict[set]
        for file in base_files:
            try:
                os.rename(file, file.replace('sber-slides/json', f'sber-slides/{set}/json'))
                img_name = os.path.split(file)[-1].replace('.json', '.png')
                os.rename(f'sber-slides/images/{img_name}', f'sber-slides/{set}/image/{img_name}')
            except:
                continue

    for set in ['train', 'dev', 'test']:
        for file in glob(f"{PATH_PREFIX}sber-slides/{set}/json/*.json"):
            f = open(file)
            data = json.load(f)
            data['meta']['split'] = set
            with open(file, 'w') as f:
                json.dump(data, f)

    # Drop invalid files
    invalid = list(map(lambda x: os.path.split(x)[-1], glob(f"{PATH_PREFIX}sber-slides/images/*")))
    if len(invalid) > 0:
        print('Invalid files remained:')
        print(invalid)

    shutil.rmtree(f"{PATH_PREFIX}sber-slides/images")
    shutil.rmtree(f"{PATH_PREFIX}sber-slides/json")

    train_images = glob(f"{PATH_PREFIX}sber-slides/train/image/*.png")
    dev_images = glob(f"{PATH_PREFIX}sber-slides/dev/image/*.png")
    test_images = glob(f"{PATH_PREFIX}sber-slides/test/image/*.png")

    get_file = lambda x: os.path.split(x)[-1]
    train_images = list(map(get_file, train_images))
    dev_images = list(map(get_file, dev_images))
    test_images = list(map(get_file, test_images))

    for img in images:
        file = os.path.split(img['file_name'])[-1]
        if file in train_images:
            img['file_name'] = f'train/image/{file}'
        elif file in dev_images:
            img['file_name'] = f'dev/image/{file}'
        elif file in test_images:
            img['file_name'] = f'test/image/{file}'
        else:
            if file not in invalid:
                print(f'File {file} is lost!')

    # Save images and categories data
    meta = {"images": images, "categories": categories}
    with open(f'sber-slides/meta.json', 'w') as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()
