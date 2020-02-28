import chainer
import chainer.computational_graph as c
from text_localization_environment import TextLocEnv
from config import CONFIG, print_config
from agent.datasets import load_dataset
from agent.custom_model import CustomModel

"""
Set arguments w/ config file (--config) or cli
:imagefile_path :boxfile_path
"""
def main():
    print_config()

    dataset = load_dataset(CONFIG['dataset'], CONFIG['dataset_path'])

    env = TextLocEnv(dataset.image_paths, dataset.bounding_boxes)
    m = CustomModel(10)
    vs = [m(env.reset())]
    g = c.build_computational_graph(vs)
    with open('graph.dot', 'w') as o:
        o.write(g.dump())

if __name__ == '__main__':
    main()
