import os
import requests
from tqdm import tqdm
import imghdr
import json
import re

val_file = 'Validation_GCC-1.1.0-Validation.tsv'

def get_val():
    os.makedirs('val_cc3m', exist_ok=True)
    
    annos = []
    with open(val_file, 'r') as f:
        lines = f.readlines()
        total = len(lines)
        del lines
        f.seek(0)
        
        idx = 0
        for _ in tqdm(range(total)):
            line = f.readline()
            splits = re.split('http', line)
            caption = splits[0]
            if len(splits) > 2:
                url = ''
                for sp in splits[1:]:
                    url = url + 'http' + sp
            else:
                url = 'http' + splits[1]
            
            response = requests.get(url)
            img_path = os.path.join('val_cc3m', 'val_' + str(idx).rjust(12, '0') + '.jpg')
            if response.status_code == 200:
                with open(img_path, 'wb') as img:
                    img.write(response.content)
                
                if imghdr.what(img_path):
                    annos.append(dict(
                        img=str(idx).rjust(12, '0'),
                        caption=caption
                    ))
                    idx += 1
                else:
                    os.remove(img_path)
                    continue
            else:
                continue
    
    print(f'valid images: {len(annos)}')
    with open('anno_val_cc3m.json', 'x') as anno:
        json.dump(annos, anno)

get_val()        
        