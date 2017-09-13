from __future__ import division
import caffe
import numpy as np
import os
import surgery
import sys
from datetime import datetime
from PIL import Image

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = 'footsegmodel1_iter_420.caffemodel'

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_mode_cpu()

solver = caffe.SGDSolver('deploy.prototxt')
raw_input("net made")
solver.net.copy_from(weights)
raw_input("net weighted")

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

#solver.step(1)
layer='score'
gt='label'
solver.test_nets[0].share_with(solver.net)
net=solver.test_nets[0]
n_cl = net.blobs[layer].channels
net.forward()
output = net.blobs[layer].data[0].argmax(0).astype(np.uint8)

output[output >= 1] = 255
im = Image.fromarray(output, mode='L')
#im.save(os.path.join('footseg12.png'))
im.show()

net.forward()
output = net.blobs[layer].data[0].argmax(0).astype(np.uint8)
output[output >= 1] = 255
im = Image.fromarray(output, mode='L')
im.show()
