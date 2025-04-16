import os
import edt
import yaml
import numpy as np
from skimage.segmentation import expand_labels
from skimage.segmentation import find_boundaries

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, MrcWriter


def generate_boundary_masks(instance_segmentation):
    return find_boundaries(instance_segmentation, connectivity=1, mode='thick', background=0)


def generate_distance_map(instance_segmentation):
    return edt.sdf(instance_segmentation, black_border=True, parallel=1)


def main(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    writer = MrcWriter(output_dtype=np.float32, overwrite=True)
    writer.set_metadata({'voxel_size': 1})
    reader = MrcReader(read_in_mem=True)


    instances_folder = cfg['instance_labels_folder']
    temp_dir = cfg['temp_dir']

    os.makedirs(os.path.join(temp_dir, 'foreground'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'distances'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'boundaries'), exist_ok=True)
    files = [x for x in os.listdir(instances_folder) if x.endswith(cfg['file_extension'])]
    if cfg['train_files'] is not None:
        if cfg['val_files'] is None:
            cfg['val_files'] = []
        files = [x for x in files if x in cfg['train_files'] or x in cfg['val_files']]

    for file in files:
        print(f'Processing tomo {file}')
        root_file_name = file.split(cfg['file_extension'])[0]
        instances = reader.read(os.path.join(instances_folder, f'{root_file_name}{cfg["file_extension"]}'))
        instances = instances.data
        instances.setflags(write=True)

        instances = expand_labels(instances, distance=1)

        foreground = (instances > 0).astype(np.int8)
        writer.set_data_array(foreground, channel_dim=None)
        writer.write(os.path.join(temp_dir, 'foreground', f'{root_file_name}{cfg["file_extension"]}'))
        boundaries = generate_boundary_masks(instances).astype(np.int16)
        writer.set_data_array(boundaries, channel_dim=None)
        writer.write(os.path.join(temp_dir, 'boundaries', f'{root_file_name}{cfg["file_extension"]}'))
        distance_map = generate_distance_map(instances)
        writer.set_data_array(distance_map, channel_dim=None)
        writer.write(os.path.join(temp_dir, 'distances', f'{root_file_name}{cfg["file_extension"]}'))


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
