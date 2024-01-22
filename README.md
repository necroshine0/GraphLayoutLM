# GraphLayoutLM

## Installation

Note that **CORDv0** dataset version is used. Should be downloaded manually or as unified .zip file. 

### Google Colab

```
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install -q 'git+https://github.com/facebookresearch/detectron2.git'

!git clone -q https://github.com/necroshine0/GraphLayoutLM
!pip install -q -r GraphLayoutLM/requirements.txt

!pip install -q gdown==v4.6.3 --no-cache-dir
!mkdir -p GraphLayoutLM/datasets && mkdir -p GraphLayoutLM/pretrained

# Base finetune
!cd GraphLayoutLM/pretrained && gdown --no-check-certificate --folder 1F593PVKVGFIfpJyRSiMZmZywZevmlKhs --quiet
# CORD dataset
!cd GraphLayoutLM/datasets && gdown 1gYX_AqrKUIqJxL1EdvoSmbyQONu4c5hF --quiet && unzip -qq CORD.zip && rm CORD.zip
```

### Windows (Conda, python 3.7)

```
conda env create -f environment.yml
conda activate graphlayoutlm
```


## Pre-trained Models
| Model               | Model Name (Path)                                                                                              | 
|---------------------|----------------------------------------------------------------------------------------------------------------|
| graphlayoutlm-base  | [graphlayoutlm-base](https://drive.google.com/drive/folders/1KV2r4crHcGoTKM7DvEIN6BWEMEdV9tIZ?usp=drive_link)  |
| graphlayoutlm-large | [graphlayoutlm-large](https://drive.google.com/drive/folders/1-zM5L34quKQwfvROvlK7UJWU6HKAGAmF?usp=drive_link) |


## Finetuning Examples

## CORD


  |Model on CORD                                                                                                                | precision | recall |    f1    | accuracy |
  |:---------------------------------------------------------------------------------------------------------------------------:|:---------:|:------:|:--------:|:--------:|
  | [graphlayout-base-finetuned-cord](https://drive.google.com/drive/folders/1F593PVKVGFIfpJyRSiMZmZywZevmlKhs?usp=drive_link)  |   0.9724  | 0.9760 |  0.9742  |  0.9813  |
  | [graphlayout-large-finetuned-cord](https://drive.google.com/drive/folders/1ZZzxG2qDnkoiADovZLovIxdhwwuqJfPc?usp=drive_link) |   0.9791  | 0.9805 |  0.9798  |  0.9839  |

### finetune

Download the model weights and move it to a new directory named "pretrained".

Download the [CORDv0](https://huggingface.co/datasets/naver-clova-ix/cord-v2) dataset and move it to a new directory named "datasets".

#### base

```
!cd GraphLayoutLM/examples && python run_cord.py --dataset_name cord \
    --do_train --do_eval \
    --model_name_or_path ../pretrained/graphlayoutlm-base-finetuned-cord  \
    --output_dir ../test/cord_base \
    --max_steps 2000 --learning_rate 5e-5
    --segment_level_layout 1 --visual_embed 1 \
    --input_size 224  --save_steps -1 \
    --evaluation_strategy steps --eval_steps 100 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 --dataloader_num_workers 8 \
    --overwrite_output_dir
```

#### large

```
!cd GraphLayoutLM/examples && python run_cord.py --dataset_name cord \
    --do_train --do_eval \
    --model_name_or_path ../pretrained/graphlayoutlm-large-finetuned-cord  \
    --output_dir ../test/cord_large \
    --max_steps 4000 --learning_rate 5e-5 \
    --segment_level_layout 1 --visual_embed 1 \
    --input_size 224 --save_steps -1 \
    --evaluation_strategy steps --eval_steps 100 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 --dataloader_num_workers 8 \
    --overwrite_output_dir
```


## Citation
Please cite our paper if the work helps you.
```
@inproceedings{li2023enhancing,
  title={Enhancing Visually-Rich Document Understanding via Layout Structure Modeling},
  author={Li, Qiwei and Li, Zuchao and Cai, Xiantao and Du, Bo and Zhao, Hai},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4513--4523},
  year={2023}
}
```


## Note

We will follow-up complement other examples.
