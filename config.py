import os 

ROOT_DIR = os.path.dirname(os.path.abspath('__file__')) 

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")

TRAIN_CSV_FILE = os.path.join(DATA_DIR, "raw", "train.csv")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "raw", "train_images")

TEST_IMAGES_DIR = os.path.join(DATA_DIR, "raw", "test_images")

STAINED_IMAGES_DIR = os.path.join(DATA_DIR, "stained", "train_images")
STAINED_IMAGES_DIR_640 = os.path.join(DATA_DIR, "stained", "train_images_640")
STAINED_IMAGES_DIR_768 = os.path.join(DATA_DIR, "stained", "train_images_768")


TRAIN_DIR_640 = os.path.join(DATA_DIR, "train_640")
TRAIN_IMAGES_DIR_640 = os.path.join(TRAIN_DIR_640, "train_images_640")
TRAIN_MASK_DIR_640 = os.path.join(TRAIN_DIR_640, "mask_640")




TRAIN_DIR_768 = os.path.join(DATA_DIR, "train_768")
TRAIN_IMAGES_DIR_768 = os.path.join(TRAIN_DIR_768, "train_images_768")
TRAIN_MASK_DIR_768 = os.path.join(TRAIN_DIR_768, "mask_768")


CLASS_TO_ID = {
    'prostate':1,
    'spleen':2,
    'lung':3,
    'kidney':4,
    'largeintestine':5   
}

MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR , "output")