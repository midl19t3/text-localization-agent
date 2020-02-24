from chainer import functions as F
from chainer import links as L
from chainer import Chain
from .resnet_group_norm import ResNet as ResNetGroupNorm


class CustomModel(Chain):
    def __init__(self, n_actions):
        super(CustomModel, self).__init__()
        with self.init_scope():
            self.resNet=L.ResNet50Layers()
            self.l1=L.Linear(2138, 1024)
            self.l2=L.Linear(1024, 1024)
            self.l3=L.Linear(1024, n_actions)

    def forward(self, x):
        image, history = x[0], x[1]
        # keep batch axis at dim 0, move x,y axes in front of color channel axis
        image = F.transpose(image, axes=(0, 3, 1, 2))
        history = F.reshape(history.astype('float32'),(-1,90))
        h1 = F.relu(self.resNet(image, layers=['pool5'])['pool5'])
        h1 = F.reshape(F.concat((h1, history), axis=1), (-1,2138))
        h2 = F.relu(self.l1(h1))
        h3 = F.relu(self.l2(h2))
        return F.relu(self.l3(h3))
