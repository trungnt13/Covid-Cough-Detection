import glob
import zipfile

import numpy as np

path = '/home/trung/Downloads/final_run/covid_data/results'

files = glob.glob(f'{path}/**/final_pri_test.zip', recursive=True)
for f in files:
  print(f)

name = 'results.csv'
all_results = []

for f in files:
  with zipfile.ZipFile(f, 'r') as f:
    data = str(f.read(f.namelist()[0]), 'utf-8').split('\n')
    header = data[0]
    ID = [i.split(',')[0] for i in data[1:]]
    x = [float(i.split(',')[1]) for i in data[1:]]
    all_results.append(np.array(x)[:, None])
all_results = np.concatenate(all_results, -1)

all_results = np.mean(all_results, axis=-1)

text = header + '\n'
for i, j in zip(ID, all_results):
  text += f'{i},{j}\n'
text = text[:-1]

with open(f'/tmp/{name}', 'w') as f:
  f.write(text)

with zipfile.ZipFile('/tmp/ensemble.zip', 'w') as f:
  f.write(f'/tmp/{name}', arcname=name)
  print('Saved!')
