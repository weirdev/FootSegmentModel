# Run net on specified inputs
import caffe
from PIL import Image
import numpy as np
import blur


def load_image(imgpath):
    """
    Load input image and preprocess for Caffe:
    - cast to float
    - switch channels RGB -> BGR
    - subtract mean
    - transpose to channel x height x width order
    """
    im = Image.open(imgpath)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]
    in_ -= (104.00699, 116.66877, 122.67892)
    in_ = in_.transpose((2, 0, 1))
    return in_


class SegNetDeploy(object):

    def __init__(self, net=caffe.Net("/home/wesley/dev/dev/caffe2/caffe/models/footsegmodel1/deploy.prototxt", caffe.TEST),
                 weightsfile='/home/wesley/dev/dev/caffe2/caffe/models/footsegmodel1/footsegmodel1train2_iter_340.caffemodel'):
        self.net = net
        self.net.copy_from(weightsfile)

    def predict(self, img):
        if isinstance(img, str):
            img = load_image(img)
        self.net.blobs["data"].reshape(1, *img.shape)
        self.net.blobs["data"].data[...] = img
        self.net.forward()
        output = self.net.blobs["score"].data[0].argmax(0).astype(np.uint8)
        output[output >= 1] = 1
        return output.astype(np.bool)


"""
if __name__ == "__main__":
    imgpaths = ["141.jpg", "142.jpg"]
    for imgpath in imgpaths:
        img = load_image(imgpath)
        net.blobs["data"].reshape(1, *img.shape)
        net.blobs["data"].data[...] = img
        net.forward()
        output = net.blobs["score"].data[0].argmax(0).astype(np.uint8)
        output[output >= 1] = 255
        im = Image.fromarray(output, mode='L')
        im.show()
        blur.blurimage(Image.open(imgpath), output).show()
"""
