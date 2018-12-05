import random

LETTERS = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
              'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
PATHS_DIR = "./paths"
TEMP_DATA_DIR = './temp_data'
DATA_DIR = './data'
IMAGE_DIR = './asl_alphabet_train'

S_COUNT = 300
SAMPLES = random.sample(range(1, 3001), S_COUNT)

