from pathlib import Path

SOURCE_DIR=Path(__file__).parent
DATA_PATH=SOURCE_DIR.parent.joinpath('data')
OUTPUT_PATH=SOURCE_DIR.parent.joinpath('output')

TB_OUT_PATH=OUTPUT_PATH.joinpath('runs') # tensorboard output
MODEL_OUT_PATH=OUTPUT_PATH.joinpath('model') # to store model params

VQA = DATA_PATH.joinpath('vqa')

VQA_OPEN_ENDED_QUESTION_TRAIN =VQA.joinpath('v2_OpenEnded_mscoco_train2017_questions.json')
VQA_OPEN_ENDED_ANSWER_TRAIN =VQA.joinpath('v2_mscoco_train2017_annotations.json')

VQA_OPEN_ENDED_QUESTION_VAL =VQA.joinpath('v2_OpenEnded_mscoco_val2017_questions.json')
VQA_OPEN_ENDED_ANSWER_VAL =VQA.joinpath('v2_mscoco_val2017_annotations.json')

VQA_OPEN_ENDED_QUESTION_TEST =VQA.joinpath('v2_OpenEnded_mscoco_test-dev2017_questions.json')


COCO_PATH = Path("/Users/xiangyuliu/sources/fiftyone_dataset_zoo/coco-2017")
CAPTION_TRAIN = COCO_PATH.joinpath("raw/captions_train2017.json")
CAPTION_VAL = COCO_PATH.joinpath("raw/captions_val2017.json")

# scale and pad the image to 640 X 640
IMAGE_SIZE = 640

# image backbone Pool dim for 640 X 640 image.
IMAGE_BACKBONE_POOL_DIM = 256 *  13 * 13

# special tokens for tokenizer
QUESTION_TOKEN='[QUESTION]'
ANSWER_TOKEN='[ANSWER]'
END_TOKEN='[END]'
EMPTY_TOKEN='[EMPTY_TOKEN]'
QA_TOKEN_DICT = {"additional_special_tokens": [QUESTION_TOKEN, ANSWER_TOKEN, END_TOKEN, EMPTY_TOKEN]}