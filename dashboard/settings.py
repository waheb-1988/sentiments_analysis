from pathlib import Path
import sys
import sys 
ROOT = str(Path.cwd().parent)
if ROOT not in sys.path: 
    sys.path.insert(0,ROOT)
import os
import random

# Function to select a random image from a directory
import os
import random

def select_random_image_from_directory(directory):
    try:
        image_files = [file for file in os.listdir(directory) if file.endswith('.png')]
        if image_files:
            random_image = random.choice(image_files)
            return os.path.join(directory, random_image)
        else:
            return None
    except Exception as e:
        print(f"Error occurred while selecting random image: {e}")
        return None

#Sources
TFIDF = 'TFIDF'
BagOfWord = 'BagOfWord'

SOURCES_LIST = [TFIDF, BagOfWord]

# Images config
IMAGES_DIR_RA= os.path.join(ROOT ,  'output' , 'rating' ,'default')
IMAGES_DIR_RATING = os.path.join(IMAGES_DIR_RA , 'default.png')
IMAGES_DIR_RAN= os.path.join(ROOT ,  'output' , 'rating' )
IMAGES_DIR_RATINGN = select_random_image_from_directory(IMAGES_DIR_RAN)

# print("IMAGES_DIR_RATING")
# print(IMAGES_DIR_RATING)
IMAGES_DIR_SE = os.path.join(ROOT , 'output' , 'sentiment','default')
IMAGES_DIR_SENTIMENT = os.path.join(IMAGES_DIR_SE , 'default.png')
IMAGES_DIR_SEN = os.path.join(ROOT , 'output' , 'sentiment')
IMAGES_DIR_SENTIMENTN = select_random_image_from_directory(IMAGES_DIR_SEN)
# IMAGES_DIR_WOR = os.path.join(ROOT , 'output' , 'wordmap')
# IMAGES_DIR_RATING = os.path.join(IMAGES_DIR_WOR , 'default.png')
IMAGES_DIR_WE = os.path.join(ROOT , 'output' , 'wordmap','default')
IMAGES_DIR_Wordmap = os.path.join(IMAGES_DIR_WE , 'default.png')
IMAGES_DIR_WEN = os.path.join(ROOT , 'output' , 'wordmap')
IMAGES_DIR_WordmapN = select_random_image_from_directory(IMAGES_DIR_WEN)
