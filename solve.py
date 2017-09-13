import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

#weights = 'fcn16s-heavy-pascal.caffemodel'
weights = 'footsegmodel1train2_iter_340.caffemodel'

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')
raw_input("net made")
solver.net.copy_from(weights)
raw_input("net weighted")

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    #score.seg_tests(solver, False, val, layer='score')
