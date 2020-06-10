from glob import glob
import os
from tqdm import tqdm


model_dirs = glob('./model/*[0-9]')
for model_dir in tqdm(model_dirs):
    os.system(f'./run.sh {model_dir} data')
