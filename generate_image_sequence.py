import os
from tqdm import tqdm
from load_agent import load_agent, create_environment

from config import CONFIG, print_config


"""
Set arguments w/ config file (--config) or cli
:gpu_id :imagefile_path :boxfile_path :agentdir_path
"""
def generate_image_sequence():
    """
    Usage:
    * Generate a dataset with exactly one image
      $ cd dataset-generator/ && python main.py -c 1
    * Generate step images for the agent that should be run
      $ python generate_image_sequence.py (params: see @click.options or run with --help flag)
    * Generate a video out of the images using ffmpeg:
      $ ffmpeg -framerate 2 -i human/%03d.png \
        -framerate 2 -i box/%03d.png \
        -filter_complex "[0:v]scale=224:-1,pad=iw+6:ih:color=white[v0];[v0][1:v]hstack=inputs=2" \
        -pix_fmt yuv420p \
        output.mp4
    """
    max_steps_per_image = 50

    env = create_environment(CONFIG['imagefile_path'], CONFIG['boxfile_path'], CONFIG['gpu_id'])
    agent = load_agent(env, CONFIG['agentdir_path'], CONFIG['gpu_id'], epsilon=0.0)

    images_human = []
    images_box = []

    obs = env.reset()

    images_human.append(env.render('human', True))
    images_box.append(env.render('box', True))

    steps_in_current_image = 1
    with tqdm(total=max_steps_per_image) as pbar:
        while steps_in_current_image <= max_steps_per_image:
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            images_human.append(env.render('human', True))
            images_box.append(env.render('box', True))
            steps_in_current_image += 1
            pbar.update(1)
            if done:
                print('Agent pulled the trigger at step ' + str(steps_in_current_image))
                break

    print('Trying to save the resulting images …')

    if not os.path.exists('human'):
        os.makedirs('human')
    if not os.path.exists('box'):
        os.makedirs('box')

    for index, image in enumerate(images_human):
        image.save('human/' + str(index).zfill(3) + '.png', 'PNG')

    for index, image in enumerate(images_box):
        image.save('box/' + str(index).zfill(3) + '.png', 'PNG')

    print('Sucessfully saved the resulting images.')

    return


if __name__ == '__main__':
    generate_image_sequence()
