# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Helper functions"""

import os
from glob import glob


# def find_model(path, epoch='latest'):
def find_model(path, epoch):
    if epoch == 'latest':
        # files = glob(os.path.join(path, '*.pth'))
        files = glob(r'output\256_shortcut1_inject0_none_hq\custom_testing\checkpoint\*.pth')
        print(files)
        file = sorted(files, key=lambda x: int(x.rsplit('.', 2)[1]))[-1]
    else:
        file = os.path.join(path, 'weights.{:d}.pth'.format(int(epoch)))
    assert os.path.exists(file), 'File not found: ' + file
    print('Find model of {} epoch: {}'.format(epoch, file))
    return file