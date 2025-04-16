import os
import yaml
import h5py
import torch
from monai.networks import one_hot
from torch.nn import Softmax, Sigmoid
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, MeanIoU
from monai.data import Dataset, list_data_collate, GridPatchDataset, ITKReader, NumpyReader
from monai.transforms import (
    Compose,
    EnsureType,
    LoadImaged,
    EnsureTyped,
    SpatialPad,
    AsDiscreted,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    EnsureChannelFirstd
)

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, PatchIter
from cryosiam.apps.dense_simsiam_semantic import load_backbone_model, load_prediction_model


def main(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    if 'trained_model' in cfg and cfg['trained_model'] is not None:
        checkpoint_path = cfg['trained_model']
    else:
        checkpoint_path = os.path.join(cfg['log_dir'], 'model', 'model_best.ckpt')
    backbone = load_backbone_model(checkpoint_path)
    prediction_model = load_prediction_model(checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    net_config = checkpoint['hyper_parameters']['config']

    test_folder = cfg['data_folder']
    labels_folder = cfg['labels_folder']
    prediction_folder = cfg['prediction_folder']
    num_classes = net_config['parameters']['network']['out_channels']
    threshold = cfg['parameters']['network']['threshold'] if 'threshold' in cfg['parameters']['network'] else 0.5
    patch_size = net_config['parameters']['data']['patch_size']
    spatial_dims = net_config['parameters']['network']['spatial_dims']
    os.makedirs(prediction_folder, exist_ok=True)
    files = cfg['test_files']

    if files is None:
        files = [x for x in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, x))]

    files = [x for x in files if os.path.exists(os.path.join(labels_folder, x))]

    test_data = []
    for idx, file in enumerate(files):
        test_data.append({'image': os.path.join(test_folder, file),
                          'labels': os.path.join(labels_folder, file),
                          'file_name': os.path.join(test_folder, file)})
    transforms = Compose(
        [
            LoadImaged(keys=['image', 'labels'], reader=MrcReader(read_in_mem=True)),
            EnsureChannelFirstd(keys=['image', 'labels'], channel_dim='no_channel'),
            ScaleIntensityRanged(keys=['image'], a_min=cfg['parameters']['data']['min'],
                                 a_max=cfg['parameters']['data']['max'], b_min=0, b_max=1, clip=True),
            NormalizeIntensityd(keys='image', subtrahend=cfg['parameters']['data']['mean'],
                                divisor=cfg['parameters']['data']['std']),
            EnsureTyped(keys=['image'], dtype=torch.float32),
            EnsureTyped(keys=['labels'], dtype=torch.int64)
        ]
    )
    pad_transform = SpatialPad(spatial_size=patch_size, method='end', mode='constant')
    if spatial_dims == 2:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0), overlap=(0, 0.5, 0.5))
    else:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0, 0), overlap=(0, 0.5, 0.5, 0.5))

    test_dataset = Dataset(data=test_data, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    print('Evaluation')
    dice_metric = DiceMetric(include_background=True, reduction="none",
                             num_classes=cfg['parameters']['network']['out_channels'])
    iou_metric = MeanIoU(include_background=True, reduction="none")
    with torch.no_grad():
        for i, test_sample in enumerate(test_loader):
            gt = test_sample['labels'][0][0]
            if cfg['eval_skip_prediction']:
                with h5py.File(os.path.join(prediction_folder,
                                            f"{os.path.basename(test_sample['file_name'][0]).split(cfg['file_extension'])[0]}_preds.h5")) as f:
                    labels_out = f['labels'][()]
                gt = one_hot(torch.unsqueeze(gt, dim=0), num_classes, dtype=torch.int, dim=0)
                labels_out = one_hot(torch.unsqueeze(torch.from_numpy(labels_out.astype('int32')), dim=0), num_classes,
                                     dtype=torch.int, dim=0)
                gt = torch.unsqueeze(gt, dim=0)
                labels_out = torch.unsqueeze(labels_out, dim=0)
                dice_metric(y_pred=labels_out, y=gt)
                iou_metric(y_pred=labels_out, y=gt)
                continue
            original_size = test_sample['image'][0][0].shape
            img = pad_transform(test_sample['image'][0])
            patch_dataset = GridPatchDataset(data=[img], patch_iter=patch_iter)
            input_size = list(img[0].shape)
            labels_out = torch.zeros(input_size, dtype=torch.int32)
            loader = DataLoader(patch_dataset, batch_size=cfg['hyper_parameters']['batch_size'], num_workers=2)
            for item in loader:
                img, coord = item[0], item[1].numpy().astype(int)
                z, _ = backbone.forward_predict(img.cuda())
                out, _ = prediction_model(z)
                if num_classes == 1:
                    out = torch.squeeze(Sigmoid()(out) > threshold, dim=1)
                else:
                    # for multiclass we apply softmax and take the class with the highest probability
                    out = Softmax(dim=1)(out)  # (B, C, D, H, W)
                    out = torch.argmax(out, dim=1, keepdim=False)  # (B, D, H, W)
                for batch_i in range(img.shape[0]):
                    c_batch = coord[batch_i][1:]
                    o_batch = out[batch_i]
                    # avoid getting patch that is outside of the original dimensions of the image
                    if c_batch[0][0] >= input_size[0] - patch_size[0] // 4 or \
                            c_batch[1][0] >= input_size[1] - patch_size[1] // 4 or \
                            (spatial_dims == 3 and c_batch[2][0] >= input_size[2] - patch_size[2] // 4):
                        continue
                    # create slices for the coordinates in the output to get only the middle of the patch
                    # and the separate cases for the first and last patch in each dimension
                    slices = tuple(
                        slice(c[0], c[1] - p // 4) if c[0] == 0 else slice(c[0] + p // 4, c[1])
                        if c[1] >= s else slice(c[0] + p // 4, c[1] - p // 4)
                        for c, s, p in zip(c_batch, input_size, patch_size))
                    # create slices to crop the patch so we only get the middle information
                    # and the separate cases for the first and last patch in each dimension
                    slices2 = tuple(
                        slice(0, 3 * p // 4) if c[0] == 0 else slice(p // 4, p - (c[1] - s))
                        if c[1] >= s else slice(p // 4, 3 * p // 4)
                        for c, s, p in zip(c_batch, input_size, patch_size))
                    labels_out[slices] = o_batch[slices2]
            labels_out = labels_out[tuple([slice(0, n) for n in original_size])]
            gt = one_hot(torch.unsqueeze(gt, dim=0), num_classes if num_classes > 1 else 2,
                         dtype=torch.int, dim=0)
            labels_out = one_hot(torch.unsqueeze(labels_out, dim=0), num_classes if num_classes > 1 else 2,
                                 dtype=torch.int, dim=0)
            gt = torch.unsqueeze(gt, dim=0)
            labels_out = torch.unsqueeze(labels_out, dim=0)
            dice_metric(y_pred=labels_out, y=gt)
            iou_metric(y_pred=labels_out, y=gt)

    global_dice_val = dice_metric.aggregate("mean").item()
    dice_vals = dice_metric.aggregate("mean_batch").cpu().detach().numpy()
    global_iou_val = iou_metric.aggregate("mean").item()
    iou_vals = iou_metric.aggregate("mean_batch").cpu().detach().numpy()
    print(f'Dice: {dice_vals.tolist()}')
    print(f'IoU: {iou_vals.tolist()}')
    print(f'Mean Dice: {global_dice_val}')
    print(f'Mean IoU: {global_iou_val}')

    import pickle
    with open(os.path.join(prediction_folder, 'ious.pkl'), 'wb') as handle:
        pickle.dump({'dice': dice_vals.tolist(), 'iou': iou_vals.tolist(),
                     'mean_dice': global_dice_val, 'mean_iou': global_iou_val},
                    handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
