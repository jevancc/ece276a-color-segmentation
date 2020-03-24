import os

l = os.listdir('./trainset')
with open('filelist.txt', 'w') as f:
    f.write('\n'.join(l))
