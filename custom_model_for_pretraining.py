from chainer import functions as F
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer import links as L
from chainer import Chain
from resnet_group_norm import ResNet as ResNetGroupNorm


class CustomModelForPretraining(Chain):
    def __init__(self, class_labels=1000):
        super(CustomModelForPretraining, self).__init__()
        with self.init_scope():
            self.resNet=ResNetGroupNorm(n_layers=18, class_labels=class_labels)

    def forward(self, image):
        image = F.reshape(image, (-1,3,224,224))
        h1 = self.resNet(image)
        return h1
