import os
from pathlib import Path

ROWS_PER_FRAME = 543  # number of landmarks per frame
ROWS_FOR_HAND = 21 * 2 # 21 landmarks each hand (from 0 to 20 inclusive)
ALL_DATA_COLUMNS = ['frame', 'row_id', 'type', 'landmark_index','x', 'y',]  # for simplicity, we ignore depth dimension 'z']

SOURCE_DIR=Path(__file__).parent
DATA_PATH=SOURCE_DIR.parent.joinpath('data')
INDEX_MAP_DATA_PATH=DATA_PATH.joinpath('sign_to_prediction_index_map.json')
