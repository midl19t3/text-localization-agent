from chainer import functions as F
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer import links as L
from chainer import Chain
from resnet_group_norm import ResNet as ResNetGroupNorm


class CustomModel(Chain):
    def __init__(self, n_actions):
        super(CustomModel, self).__init__()
        with self.init_scope():
            self.resNet=ResNetGroupNorm(n_layers=18)
            self.l1=L.Linear(602, 1024)
            self.l2=L.Linear(1024, 1024)
            self.l3=L.Linear(1024, n_actions)

    def forward(self, x):
        image, history = x[0], x[1]
        image = F.reshape(image, (-1,3,224,224))
        history = F.reshape(history.astype('float32'),(-1,90))
        h1 = self.resNet(image)

        # pooling as done here: https://github.com/chainer/chainer/blob/v6.0.0/chainer/links/model/vision/resnet.py#L655
        n, channel, rows, cols = h1.shape
        h1 = average_pooling_2d(h1, (rows, cols), stride=1)
        h1 = F.reshape(h1, (n, channel))

        h1 = F.relu(h1)
        h1 = F.reshape(F.concat((h1, history), axis=1), (-1,602))
        h2 = F.relu(self.l1(h1))
        h3 = F.relu(self.l2(h2))
        return F.relu(self.l3(h3))
