import os 

ROOT_DIR = os.path.dirname(os.path.abspath('__file__')) 

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")

TRAIN_CSV_FILE = os.path.join(DATA_DIR, "raw", "train.csv")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "raw", "train_images")

TEST_IMAGES_DIR = ""

CLASS_TO_ID = {
    'prostate':1,
    'spleen':2,
    'lung':3,
    'kidney':4,
    'largeintestine':5   
}

MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR , "output")