import numpy as np

import torch
from torch.autograd import Variable

from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D


def get_sport_clip(clip_name, verbose=True):

    clip = sorted(glob(join('data', clip_name, '*.png')))
    clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in clip])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)


def read_labels_from_file(filepath):
    with open(filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def main():

    X = get_sport_clip('roger')
    X = Variable(X)
    X = X.cuda()

    # get network pretrained model
    net = C3D()
    net.load_state_dict(torch.load('c3d.pickle'))
    net.cuda()
    net.eval()

    prediction = net(X)
    prediction = prediction.data.cpu().numpy()

    labels = read_labels_from_file('labels.txt')

    top_inds = prediction[0].argsort()[::-1][:5]  # reverse sort and take five largest items
    print('\nTop 5:')
    for i in top_inds:
        print('{:.5f} {}'.format(prediction[0][i], labels[i]))


if __name__ == '__main__':
    main()
