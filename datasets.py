import json
import os
import numpy as np
import scipy.io as sio
from PIL import Image
from pathlib import Path
from abc import ABC, abstractmethod

class Dataset(ABC):
    id = NotImplemented

    def __init__(self, dataset_path):
        self.dataset_path = os.path.abspath(dataset_path)
        self.image_paths = []
        self.bounding_boxes = []

    @abstractmethod
    def load(self):
        pass

    def get(self, index, as_image=True):
        image = self.image_paths[index]
        if as_image:
            image = Image.open(image)
        true_bboxes = self.true_bboxes[index]
        return image, true_bboxes

    def get_image_name(self, index):
        image_path, _ = self.get(index, as_image=False)
        image_fname = Path(image_path).name
        image_name = image_fname.split('.')[0]
        return image_name

    def random_sample(self, as_image=True):
        random_index = np.random.randint(len(self.image_paths))
        return self.get(random_index, as_image=as_image)

    def __len__(self):
        return len(self.image_paths)

class SimpleDataset(Dataset):
    id = 'simple'

    def load(self):
        image_locations_file = os.path.join(self.dataset_path, 'image_locations.txt')
        relative_image_paths = np.loadtxt(image_locations_file, dtype=str)
        self.image_paths = [os.path.join(self.dataset_path, image_path) for image_path in relative_image_paths]
        
        bounding_boxes_file = os.path.join(self.dataset_path, 'bounding_boxes.npy')
        self.bounding_boxes = np.load(bounding_boxes_file, allow_pickle=True)

class SignDataset(Dataset):
    id = 'sign'

    def load(self):
        config_file_path = os.path.join(self.dataset_path, 'training.json')

        absolute_image_paths = []
        bounding_boxes = []

        with open(config_file_path) as f:
            data = json.load(f)
            for image in data:
                absolute_path = os.path.join(self.dataset_path, image['file_name'])
                if not os.path.exists(absolute_path): continue
                absolute_image_paths.append(absolute_path)
                bounding_boxes.append(list(image['bounding_boxes']))

        self.image_paths = absolute_image_paths
        self.bounding_boxes = bounding_boxes

class SynthTextDataset(Dataset):
    id = 'synthtext'

    def data(self):
        """
        mat file structure:
        {
            'imnames': [[path_1, path_2, ...]]
            'wordBB': [
                # image 1
                [
                    [
                        [x0_txt0, x0_txt1, ...],
                        [x1_txt0, x1_txt2, ...],
                        [x2_txt0, x2_txt2, ...],
                        [x3_txt0, x3_txt2, ...],
                    ],
                    [
                        [y0_txt0, y0_txt1, ...],
                        [y1_txt0, y1_txt2, ...],
                        [y2_txt0, y2_txt2, ...],
                        [y3_txt0, y3_txt2, ...],
                    ],
                    # but if there is only one text the format is:
                    [x0, x1, x2, x3],
                    [y0, y1, y2, y3],
                ],
                # image 2
                [
                    ...
                ],
                ...
            ]
        }

        corner points (clockwise):
        x0|y0 ---- x1|y1
          .          .
          .          .
        x3|y3 ---- x2|y2
        """

        mat_file = os.path.join(self.dataset_path, 'gt.mat')
        mat = sio.loadmat(mat_file)
        relative_image_paths = [arr[0] for arr in mat['imnames'][0]]
        absolute_image_paths = [os.path.join(self.dataset_path, image_path) for image_path in relative_image_paths]
        bounding_boxes = []

        print(f"Processing bounding boxes of the dataset...")

        for i, image_bboxes in enumerate(mat['wordBB'][0]):
            if i % 100000 == 99999:
                print(f"Processed bounding boxes of {i + 1} images.")

            output_bboxes = []

            if hasattr(image_bboxes[0][0], "__len__"):
                for index in range(len(image_bboxes[0][0])):
                    x0 = image_bboxes[0][0][index]
                    x1 = image_bboxes[0][1][index]
                    x2 = image_bboxes[0][2][index]
                    x3 = image_bboxes[0][3][index]
                    y0 = image_bboxes[1][0][index]
                    y1 = image_bboxes[1][1][index]
                    y2 = image_bboxes[1][2][index]
                    y3 = image_bboxes[1][3][index]
                    output_bboxes.append([int(min(x0, x3)), int(min(y0, y1)), int(max(x1, x2)), int(max(y2, y3))])
            else:
                x0, x1, x2, x3 = image_bboxes[0]
                y0, x1, x2, x3 = image_bboxes[1]
                output_bboxes.append([int(min(x0, x3)), int(min(y0, y1)), int(max(x1, x2)), int(max(y2, y3))])
            
            bounding_boxes.append(output_bboxes)

        self.bounding_boxes = bounding_boxes
        self.image_paths = absolute_image_paths
        
def load_dataset(id, path):
    datasets = [SimpleDataset, SignDataset, SynthTextDataset]
    dataset = {Dataset.id: Dataset for Dataset in datasets}[id](path)
    return dataset
  