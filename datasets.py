import json
import os
import numpy as np
import scipy.io as sio

class Dataset:
    id = 'noop'

    def __init__(self, dataset_path):
        self.dataset_path = os.path.abspath(dataset_path)

    def data(self):
        pass  

class SimpleDataset(Dataset):
    id = 'simple'

    def data(self):
        image_locations_file = os.path.join(self.dataset_path, 'image_locations.txt')
        relative_image_paths = np.loadtxt(image_locations_file, dtype=str)
        absolute_image_paths = [os.path.join(self.dataset_path, image_path) for image_path in relative_image_paths]
        
        bounding_boxes_file = os.path.join(self.dataset_path, 'bounding_boxes.npy')
        bounding_boxes = np.load(bounding_boxes_file, allow_pickle=True)

        return absolute_image_paths, bounding_boxes

class SignDataset(Dataset):
    id = 'sign'

    def data(self):
        config_file_path = os.path.join(self.dataset_path, 'training.json')

        absolute_image_paths = []
        bounding_boxes = []

        with open(config_file_path) as f:
            data = json.load(f)
            for image in data:
                absolute_path = os.path.join(self.dataset_path, image['file_name'])
                if not os.path.exists(absolute_path): continue
                absolute_paths.append(absolute_path)
                bboxes.append(list(image['bounding_boxes']))

        return absolute_image_paths, bounding_boxes

class SynthTextDataset(Dataset):
    id = 'synthtext'

    def data(self):
        mat_file = os.path.join(self.dataset_path, 'gt.mat')
        mat = sio.loadmat(mat_file)
        relative_image_paths = [arr[0] for arr in mat['imnames'][0]]
        absolute_image_paths = [os.path.join(self.dataset_path, image_path) for image_path in relative_image_paths]
        bounding_boxes = mat['wordBB'][0]
        return absolute_image_paths, bounding_boxes
        
def get_dataset(id):
    datasets = [SimpleDataset, SignDataset, SynthTextDataset]
    return {Dataset.id: Dataset for Dataset in datasets}[id]
  