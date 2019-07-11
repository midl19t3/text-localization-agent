import argparse
import json
import math
import random
from datetime import datetime

import numpy as np
import chainer
from chainer import backend
from chainer import backends
from chainer import datasets
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from chainer.backends.cuda import cupy

from PIL import Image

from custom_model_for_pretraining import CustomModelForPretraining

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--learning-rate', default=0.01, type=float)
    parser.add_argument('--max-epoch', default=100, type=int)  # TODO: what's a senseful number?
    parser.add_argument('--dataset-part-size', default=-1, type=int)

    args = parser.parse_args()

    print(args)
    return args

@chainer.dataset.converter()
def resize_images_from_batch(batch, device):

    def convert_single_example(example):
        # import pdb;pdb.set_trace()
        image_array = example[0]
        n_channels, height, width = image_array.shape
        if n_channels != 3:
            # repeat single greyscale color channel three times to get RGB representation
            image_array = np.repeat(image_array, 3, axis=0)
        # print('current shape', (n_channels, height, width))
        reshaped = cupy.transpose(image_array, (1, 2, 0))
        # print('reshaped shape', reshaped.shape)
        im = Image.fromarray(reshaped)
        import pdb;pdb.set_trace()
        # im.save('/home/padl19t1/example_before.png', 'PNG')
        if im.mode != 'RGB':
            im = im.convert('RGB')

        im = im.resize((224, 224), Image.LANCZOS)
        # print('new image size', im.size)
        # im.save('/home/padl19t1/example_after.png', 'PNG')
        as_np_array = cupy.asarray(im, dtype=cupy.float32)
        try:
            assert as_np_array.shape == (224, 224, 3)
        except Exception as e:
            import pdb;pdb.set_trace()
        # print('shape is', as_np_array.shape)
        # import pdb;pdb.set_trace()
        return (as_np_array, cupy.asarray(example[1]))

    # batch = cupy.apply_along_axis(convert_single_example, axis=1, arr=batch)
    # import pdb;pdb.set_trace()
    new_batch = []
    for example in batch:
        # import pdb;pdb.set_trace()
        new_batch.append(convert_single_example(example))

    # import pdb;pdb.set_trace()
    batch = new_batch
    assert type(batch) == list
    assert len(batch) == 128
    for elem in batch:
        try:
            assert type(elem) == tuple
            assert type(elem[0]) == cupy.ndarray
            assert elem[0].dtype == cupy.float32
            assert elem[0].shape == (224, 224, 3)
            assert type(elem[1]) == cupy.ndarray
            assert elem[1].dtype == cupy.int32
            assert elem[1].shape == ()
            assert type(np.asscalar(elem[1])) == int
        except Exception as e: 
            import pdb;pdb.set_trace()

    # call default StandardUpdater converter
    batch = chainer.dataset.concat_examples(batch, device)

    return device.send(batch)

if __name__ == '__main__':
    args = parse_args()

    mini_batch_size = 128

    with open('/data/common/imagenet/train_gt.json', 'r') as f:
        train_image_class_mapping = json.load(f)

    random.seed(42)  # make sure shuffle() always yields the same order
    random.shuffle(train_image_class_mapping)
    if args.dataset_part_size != -1:
        train_image_class_mapping = train_image_class_mapping[:args.dataset_part_size]
    print(train_image_class_mapping[:3])

    dataset_pairs = [(example['file_name'], example['class']) for example in train_image_class_mapping]

    dataset = datasets.LabeledImageDataset(dataset_pairs, root='/data/common/imagenet', dtype=cupy.uint8)

    split_index = int(len(dataset) / 10)
    test = datasets.SubDataset(dataset, start=0, finish=split_index)
    train = datasets.SubDataset(dataset, start=split_index, finish=len(dataset))

    print('len(dataset)', len(dataset))
    print('len(train)', len(train))
    print('len(test)', len(test))

    # hotencoding!


    # VAlIDATION:
    # ILSVRC2012_validation_ground_truth.txt has classes of files in val/ in the order

    # TRAIN

    # TEST
    # there seems to be no corresponding file for test/ ?!

    # (keep random seed the same)

    # 1. load necessary files, limit at 15GB




    # 2. load images from paths and convert to right representation
    # image = Image(path)
    # if image.mode != 'RGB':
    #     image = image.convert('RGB')
    # np.array(image)

    # image.resize((224, 224), LANCZOS)

    # def compute_state(self):
    #     warped = self.get_warped_bbox_contents()
    #     return (np.array(warped, dtype=np.float32), np.array(self.history))




    # 3. build actual train, test by splitting train dataset




    # let train, test be lists of tuples (x, y)
    train_iter = iterators.SerialIterator(train, mini_batch_size)
    test_iter = iterators.SerialIterator(test, mini_batch_size, repeat=False, shuffle=False)

    resNetPretrainCustomModel = CustomModelForPretraining()

    if args.gpu >= 0:
        resNetPretrainCustomModel.to_gpu(args.gpu)

    # resNetPretrainCustomModel contains linear layer as last layer that brings it down to n classes (inside resnet base model)
    # no activation function for last layer because apparently, Classifier applies softmax for us
    model = L.Classifier(resNetPretrainCustomModel)

    optimizer = optimizers.MomentumSGD(lr=args.learning_rate, momentum=0.9)
    optimizer.setup(model)

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=resize_images_from_batch)

    trainer = training.Trainer(updater, (args.max_epoch, 'epoch'), out='imagenet_result')

    # extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.DumpGraph('main/loss'))

    trainer.run()

    print('Training run finished')

    # TODO: test this end to end with loading of pretrained model into RL cycle before training for large number of epochs!
    serializers.save_npz(f'/data/padl19t1/pretrained_resnet_{str(datetime.now())}', resNetPretrainCustomModel.resNet)
