from os import listdir
from os.path import isfile, join
import numpy as np
onlytxt = [ f for f in listdir('/data/datasets/CTA/pulses') if isfile(join('/data/datasets/CTA/pulses', f)) and f.endswith('.txt')]
# read them all
pes = []
for f in onlytxt:
    file_txt = open(join('/data/datasets/CTA/pulses', f),'r')
    if int(f.split('pe')[1].split('.txt')[0])==6:continue
    pes.append(int(f.split('pe')[1].split('.txt')[0]))

pes.sort()
print(pes,pes[20])
templates = []
axes = []
for pe in pes:
    file_txt = open('/data/datasets/CTA/pulses/pe'+str(pe).zfill(5)+'.txt', 'r')
    templates.append([])
    axes.append([])
    firstline = True
    for l in file_txt.readlines():
        axes[-1].append(int(l.split('\t')[0]))
        templates[-1].append(float(l.split('\t')[1]))
# build npz
sample_array = np.array(axes[-1],dtype=np.int)
pe_array = np.array(pes,dtype=np.int)
template = np.zeros((pe_array.shape+sample_array.shape))
for i,pe in enumerate(pe_array):
    for j,sample in enumerate(sample_array):
        template[i,j]=templates[i][j]

np.savez_compressed('/data/datasets/CTA/pulses/templates_input.npz',
                    templates=template,
                    samples=sample_array,
                    pes = pe_array)