from pathlib import Path

SOURCE_DIR=Path(__file__).parent
DATA_PATH=SOURCE_DIR.parent.joinpath('data')
FLICKR30k_DIR=DATA_PATH.joinpath('flickr30k')
FLICKR30k_IMAGE_DIR=FLICKR30k_DIR.joinpath('flickr30k-images')
FLICKR30k_ANNOTATION_DIR=FLICKR30k_DIR.joinpath('annotations/Annotations')
FLICKR30k_SENTENCE_DIR=FLICKR30k_DIR.joinpath('annotations/Sentences')

FLICKR30k_TRAIN = FLICKR30k_DIR.joinpath("train.txt")
FLICKR30k_VAL= FLICKR30k_DIR.joinpath("val.txt")
FLICKR30k_TEST= FLICKR30k_DIR.joinpath("test.txt")




COCO_PATH = Path("/Users/xiangyuliu/sources/fiftyone_dataset_zoo/coco-2017")
CAPTION_TRAIN = COCO_PATH.joinpath("raw/captions_train2017.json")
CAPTION_VAL = COCO_PATH.joinpath("raw/captions_val2017.json")