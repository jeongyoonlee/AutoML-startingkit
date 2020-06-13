from glob import glob
import os
from tqdm import tqdm


model_dirs = sorted(glob('./model/*_*[0-9]'))
for model_dir in tqdm(model_dirs):
    predict_file = os.path.join('./build/', os.path.basename(model_dir), 'E_test4.predict')
    if not os.path.exists(predict_file):
        os.system(f'./run.sh {model_dir} data')
    else:
        print(f'skipping {model_dir}')
