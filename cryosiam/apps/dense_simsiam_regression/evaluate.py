import os
import yaml
import torch
import mrcfile
import numpy as np
from torch.utils.data import DataLoader
from monai.metrics import PSNRMetric
from monai.metrics.regression import SSIMMetric
from monai.data import Dataset, list_data_collate, GridPatchDataset, NumpyReader
from monai.transforms import (
    Compose,
    SpatialPad,
    LoadImaged,
    EnsureType,
    EnsureTyped,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    EnsureChannelFirstd
)

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, PatchIter
from cryosiam.apps.dense_simsiam_regression import load_backbone_model, load_prediction_model


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
    gt_folder = cfg['gt_folder']
    prediction_folder = cfg['prediction_folder']
    patch_size = net_config['parameters']['data']['patch_size']
    spatial_dims = net_config['parameters']['network']['spatial_dims']
    num_output_channels = net_config['parameters']['network']['n_output_channels']
    os.makedirs(prediction_folder, exist_ok=True)
    files = cfg['test_files']

    if files is None:
        files = [x for x in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, x))]
    test_data = []
    for idx, file in enumerate(files):
        test_data.append({'image': os.path.join(test_folder, file),
                          'gt': os.path.join(gt_folder, file),
                          'file_name': os.path.join(test_folder, file)})
    reader = MrcReader(read_in_mem=True)
    transforms = Compose(
        [
            LoadImaged(keys=['image', 'gt'], reader=reader),
            EnsureChannelFirstd(keys=['image', 'gt'], channel_dim='no_channel'),
            ScaleIntensityRanged(keys=['image'], a_min=cfg['parameters']['data']['min'],
                                 a_max=cfg['parameters']['data']['max'], b_min=0, b_max=1, clip=True),
            NormalizeIntensityd(keys='image', subtrahend=cfg['parameters']['data']['mean'],
                                divisor=cfg['parameters']['data']['std']),
            EnsureTyped(keys=['image', 'gt'], dtype=torch.float32)
        ]
    )
    pad_transform = SpatialPad(spatial_size=patch_size, method='end', mode='constant')
    post_pred = Compose([EnsureType('numpy', dtype=np.float32, device=torch.device('cpu'))])

    if spatial_dims == 2:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0), overlap=(0, 0.5, 0.5))
    else:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0, 0), overlap=(0, 0.5, 0.5, 0.5))

    test_dataset = Dataset(data=test_data, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    print('Evaluation')
    psnr_metric = PSNRMetric(max_val=1, reduction="none")
    ssim_metric = SSIMMetric(spatial_dims=spatial_dims, reduction="none")
    with torch.no_grad():
        for i, test_sample in enumerate(test_loader):
            ssim_metric = SSIMMetric(spatial_dims=spatial_dims, reduction='none')
            psnr_metric = PSNRMetric(max_val=1, reduction='none')
            gt = test_sample['gt'][0][0]
            if cfg['eval_skip_prediction']:
                gt_out = mrcfile.open(os.path.join(prediction_folder,
                                                   f"{os.path.basename(test_sample['file_name'][0]).split(cfg['file_extension'])[0]}.mrc")).data
                gt_out = torch.from_numpy(gt_out.astype('float32'))
                gt = torch.unsqueeze(torch.unsqueeze(gt, dim=0), dim=0)
                gt_out = torch.unsqueeze(torch.unsqueeze(gt_out, dim=0), dim=0)
                psnr_metric(y_pred=gt_out, y=gt)
                ssim_metric(y_pred=gt_out, y=gt)

                ssim_val = ssim_metric.aggregate("mean").item()
                psnr_val = psnr_metric.aggregate("mean").item()

                print(f'Filename: {test_data[i]["file_name"]}')
                print(f'SSIM: {ssim_val}')
                print(f'PSNR: {psnr_val}')
                continue
            original_size = test_sample['image'][0][0].shape
            img = pad_transform(test_sample['image'][0])
            patch_dataset = GridPatchDataset(data=[img], patch_iter=patch_iter)
            input_size = list(img[0].shape)
            preds_out = np.zeros([num_output_channels] + input_size, dtype=np.float32)
            loader = DataLoader(patch_dataset, batch_size=cfg['hyper_parameters']['batch_size'], num_workers=2)
            for item in loader:
                img, coord = item[0], item[1].numpy().astype(int)
                z, _ = backbone.forward_predict(img.cuda())
                out = prediction_model(z)
                out = post_pred(out)
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
                    preds_out[(slice(0, num_output_channels),) + slices] = o_batch[(slice(0, num_output_channels),)
                                                                                   + slices2]

            preds_out = preds_out[(slice(0, num_output_channels),) + tuple([slice(0, n) for n in original_size])]

            if cfg['scale_prediction']:
                preds_out = (preds_out - preds_out.min()) / (preds_out.max() - preds_out.min())

            gt = torch.unsqueeze(torch.unsqueeze(gt, dim=0), dim=0)
            preds_out = torch.unsqueeze(torch.from_numpy(preds_out), dim=0)
            psnr_metric(y_pred=preds_out, y=gt)
            ssim_metric(y_pred=preds_out, y=gt)

    mean_psnr_val = psnr_metric.aggregate("mean").item()
    psnr_vals = psnr_metric.aggregate("mean_batch").cpu().detach().numpy()
    mean_ssim_val = ssim_metric.aggregate("mean").item()
    ssim_vals = ssim_metric.aggregate("mean_batch").cpu().detach().numpy()
    print(f'PSNR: {psnr_vals}')
    print(f'Mean PSNR: {mean_psnr_val}')
    print(f'SSIM: {ssim_vals}')
    print(f'Mean SSIM: {mean_ssim_val}')

    import pickle
    with open(os.path.join(prediction_folder, 'metrics.pkl'), 'wb') as handle:
        pickle.dump({'psnr_vals': psnr_vals, 'ssim_vals': ssim_vals,
                     'mean_psnr': mean_psnr_val, 'mean_ssim': mean_ssim_val},
                    handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
