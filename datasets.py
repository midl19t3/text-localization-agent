import json
import os
import numpy as np

class Dataset:
    id = 'noop'

    def __init__(self, config):
        config_prefix = f"{self.id}_dataset_"
        self.config = {k.replace(config_prefix, ''): v for k, v in config.items() if config_prefix in k}

    def data(self):
        pass  

class SimpleDataset(Dataset):
    id = 'simple'

    def data(self):
        image_locations_file_path = self.config['image_locations']
        dataset_base_path = os.path.dirname(image_locations_file_path)
        
        relative_image_paths = np.loadtxt(image_locations_file_path, dtype=str)
        absolute_image_paths = [dataset_base_path + i.strip('.') for i in relative_image_paths]
        
        bounding_boxes = np.load(self.config['bounding_boxes'], allow_pickle=True)

        return absolute_image_paths, bounding_boxes

class SignDataset(Dataset):
    id = 'sign'

    def data(self):
        config_file_path = self.config['config']
        dataset_base_path = os.path.dirname(config_file_path)

        absolute_image_paths = []
        bounding_boxes = []

        with open(config_file_path) as f:
            data = json.load(f)
            for image in data:
                absolute_path = os.path.join(dataset_base_path, image['file_name'])
                if not os.path.exists(absolute_path): continue
                absolute_paths.append(absolute_path)
                bboxes.append(list(image['bounding_boxes']))

        return absolute_image_paths, bounding_boxes

class SynthDataset(Dataset):
    id = 'synth'

    def data(self):
        pass

def get_dataset(id):
    datasets = [SimpleDataset, SignDataset, SynthDataset]
    return {Dataset.id: Dataset for Dataset in datasets}[id]