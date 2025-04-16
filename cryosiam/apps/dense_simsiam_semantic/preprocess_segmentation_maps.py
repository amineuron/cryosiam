import os
import edt
import yaml
import numpy as np

from cryosiam.data import MrcReader
from cryosiam.utils import parser_helper


def generate_distance_map(semantic_segmentation, num_classes=1):
    dist = np.zeros((num_classes,) + semantic_segmentation.shape)
    for i in range(num_classes):
        dist[i] = edt.sdf(semantic_segmentation == i, black_border=True, parallel=1)
    return dist


def main(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    reader = MrcReader(read_in_mem=True)

    labels_folder = cfg['labels_folder']
    temp_dir = cfg['temp_dir']
    out_channels = cfg['parameters']['network']['out_channels']

    os.makedirs(os.path.join(temp_dir, 'distances'), exist_ok=True)
    files = [x for x in os.listdir(labels_folder) if x.endswith(cfg['file_extension'])]
    if cfg['train_files'] is not None:
        if cfg['val_files'] is None:
            cfg['val_files'] = []
        files = [x for x in files if x in cfg['train_files'] or x in cfg['val_files']]

    for file in files:
        print(f'Processing tomo {file}')
        root_file_name = file.split(cfg['file_extension'])[0]
        labels = reader.read(os.path.join(labels_folder, f'{root_file_name}{cfg["file_extension"]}'))
        labels = labels.data
        labels.setflags(write=True)

        distance_map = generate_distance_map(labels, num_classes=out_channels).astype(np.float32)
        np.savez_compressed(os.path.join(temp_dir, 'distances', f'{root_file_name}.npz'), data=distance_map)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
