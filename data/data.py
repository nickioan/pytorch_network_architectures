import pathlib
import random

import h5py
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk

from .utils import augment, get_patch


class CosmosData(Dataset):

    def __init__(self, root, patch=None):
        super(CosmosData, self).__init__()

        self.root = pathlib.Path(root)
        if patch is not None:
            self.root = self.root.joinpath('{}'.format(patch))

        self.files = sorted(list(self.root.glob('*.mat')))

    def loader(self, file):
        try:
            mat = loadmat(file)
        except Exception as e:
            print(file)
            raise e
        return mat

    def transform(self, x, y):
        raise NotImplementedError

    def __getitem__(self, i):
        mat = self.loader(self.files[i])

        fl = np.ascontiguousarray(mat['fl'].transpose(2,1,0))
        cosmos = np.ascontiguousarray(mat['cosmos'].transpose(2,1,0))
        mask = np.ascontiguousarray(mat['mask'].transpose(2,1,0))
        D = np.ascontiguousarray(mat['D'].transpose(2,1,0))

        return fl, cosmos, mask, D

    def __len__(self):
        return len(self.files)


class HomodyneData(Dataset):

    def __init__(self, root, patch=None, augment=False, mode='test', suffix=None):
        super(HomodyneData, self).__init__()

        root = pathlib.Path(root)
        if mode == 'train':
            self.root = root.joinpath('train')
        elif mode == 'val':
            self.root = root.joinpath('val')
        elif mode == 'test':
            self.root = root.joinpath('test')
        else:
            raise ValueError('mode must be one of train, val, test')

        if suffix is not None:
            self.root = pathlib.Path(str(self.root) + suffix)

        self.patch = patch
        self.augment = augment

        self.files = sorted(list(self.root.glob('*.mat')))

    def loader(self, file):
        try:
            with h5py.File(file, 'r') as data:
                x = np.ascontiguousarray(data['hfl'], dtype=np.float32)
                y = np.ascontiguousarray(data['fl'], dtype=np.float32)
                m = np.ascontiguousarray(data['mask'], dtype=np.bool)
        except Exception as e:
            print(file)
            raise e

        return x, y, m

    def transform(self, x, y, m):
        x = th.from_numpy(x)
        y = th.from_numpy(y)
        m = th.from_numpy(m)

        if self.patch is not None:
            x, y, m = get_patch(x, y, m, patch=self.patch)

        if self.augment:
            x, y, m = augment(x, y, m)

        return x, y, m

    def __getitem__(self, i):
        file = self.files[i]
        x, y, m = self.loader(file)
        x, y, m = self.transform(x, y, m)
        return {'x': x, 'y': y, 'm': m}

    def __len__(self):
        return len(self.files)

class MicrobleedData(Dataset):

    def __init__(self, root, patch=None, augment=False, mode='test', suffix=None):
        super(MicrobleedData, self).__init__()

        root = pathlib.Path(root)
        if mode == 'train':
            self.root_img_qsm = root.joinpath('train/images/qsm')
            self.root_img_lf = root.joinpath('train/images/localfield')
            self.root_label = root.joinpath('train/labels')
            self.root_mask = root.joinpath('train/masks')
        elif mode == 'val':
            self.root_img_qsm = root.joinpath('validation/images/qsm')
            self.root_img_lf = root.joinpath('validation/images/localfield')
            self.root_label = root.joinpath('validation/labels')
            self.root_mask = root.joinpath('validation/masks')
        elif mode == 'test':
            self.root_img_qsm = root.joinpath('test/images/qsm')
            self.root_img_lf = root.joinpath('test/images/localfield')
            self.root_label = root.joinpath('test/labels')
            self.root_mask = root.joinpath('test/masks')
        else:
            raise ValueError('mode must be one of train, val, test')

        # if suffix is not None:
        #     self.root_img = pathlib.Path(str(self.root_img) + suffix)
        #     self.root_mask = pathlib.Path(str(self.root_mask) + suffix)

        self.patch = patch
        self.augment = augment

        self.files_img_qsm = sorted(self.root_img_qsm.glob('*.nii.gz'))
        self.files_img_lf = sorted(self.root_img_lf.glob('*.nii.gz'))
        self.files_label = sorted(self.root_label.glob('*.nii.gz'))
        self.files_mask = sorted(self.root_mask.glob('*.nii.gz'))
        self.files = list(zip(self.files_img_qsm,self.files_img_lf,self.files_label,self.files_mask))


    def loader(self, my_file):
        try:
            x_qsm = sitk.ReadImage(my_file[0].absolute().as_posix())
            x_lf  = sitk.ReadImage(my_file[1].absolute().as_posix())
            y     =  sitk.ReadImage(my_file[2].absolute().as_posix())
            m     =  sitk.ReadImage(my_file[3].absolute().as_posix())
        except Exception as e:
            print(my_file)
            raise e

        return x_qsm,x_lf, y, m
    
    def resample(self,volume,is_mask):
        dsampled_vsize = (0.391,0.391,1.0)
        #dsampled_vsize = (0.45,0.45,1.2)
        size = volume.GetSize()
        vsize = volume.GetSpacing()
        vratio = [a/b for a,b in zip(dsampled_vsize,vsize)]
        new_size = [int(a/b) for a,b in zip(size,vratio)]
            
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(dsampled_vsize)
        resample.SetSize(new_size)
        resample.SetOutputDirection(volume.GetDirection())
        resample.SetOutputOrigin(volume.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(volume.GetPixelIDValue())

        if is_mask:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
        
        return resample.Execute(volume)

    def crop(self,x_qsm,x_lf,y,m):
        def first_nonzero(arr, axis, invalid_val=0):
            mask = arr!=0
            return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

        def last_nonzero(arr, axis, invalid_val=0):
            mask = arr!=0
            val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
            return np.where(mask.any(axis=axis), val, invalid_val)
        
        yx = first_nonzero(m,0)
        zx = first_nonzero(m,1)

        x_columns = first_nonzero(yx,0)
        y_columns = first_nonzero(yx,1)
        z_columns = first_nonzero(zx,1)

        ix = first_nonzero(x_columns,0)
        jx = last_nonzero(x_columns,0)

        iy = first_nonzero(y_columns,0)
        jy = last_nonzero(y_columns,0)

        iz = first_nonzero(z_columns,0)
        jz = last_nonzero(z_columns,0)

        return x_qsm[iz:jz,iy:jy,ix:jx] , x_lf[iz:jz,iy:jy,ix:jx] , y[iz:jz,iy:jy,ix:jx], m[iz:jz,iy:jy,ix:jx]

        

    def transform(self, x_qsm, x_lf, y, m):

        # x = self.resample(x,is_mask=False)
        # y = self.resample(y,is_mask=True)
        # m = self.resample(m,is_mask=True)

        x_qsm = sitk.GetArrayFromImage(x_qsm)
        x_lf = sitk.GetArrayFromImage(x_lf)
        y = sitk.GetArrayFromImage(y)
        m = sitk.GetArrayFromImage(m)

        x_qsm,x_lf,y,m = self.crop(x_qsm,x_lf,y,m)
        
        background = 1.0 - y

        x_qsm = np.expand_dims(x_qsm,0)
        x_lf = np.expand_dims(x_lf,0)

        y = np.expand_dims(y,0)
        background = np.expand_dims(background,0)

        m = np.expand_dims(m,0)
        x_qsm = th.from_numpy(x_qsm)
        x_lf = th.from_numpy(x_lf)
        y = th.from_numpy(y)
        background = th.from_numpy(background)
        m = th.from_numpy(m)

        y = th.cat((background,y),0)
        x = th.cat((x_qsm,x_lf),0)
        #x = x_qsm
        if self.patch is not None:
            x, y, m = get_patch(x, y, m , patch=self.patch)
        
        # if self.augment:
        #     x, y = augment(x, y)
        #print("TRANSFORM")    

        return x, y , m

    def __getitem__(self, i):
        my_file = self.files[i]
        x_qsm, x_lf, y, m = self.loader(my_file)
        n = my_file[0].name
        x, y, m= self.transform(x_qsm , x_lf, y, m)
        return {'x': x, 'y': y, 'm': m, 'n': n}

    def __len__(self):
        return len(self.files)


def create_dataloader(args, test=False):
    if test:
        return {
            'test' : _data_loader(args, 'test')
        }

    else:
        return {
            'train': _data_loader(args, 'train'),
            'val': _data_loader(args, 'val')
        }


def create_dataset(args, mode='test'):
    if mode == 'train':
        patch = args.patch
        augment = not args.no_augment

    else:
        patch = None
        augment = False

    return MicrobleedData(
        root = args.datadir,
        patch = patch,
        augment = augment,
        mode = mode,
        suffix = args.suffix
    )


def _data_loader(args, mode='test'):
    dataset = create_dataset(args, mode)

    if mode == 'train':
        batch_size = args.batch_size
        shuffle = True
        drop_last = True
    else:
        batch_size = 1
        shuffle = False
        drop_last = False

    return DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = args.num_workers,
        shuffle = shuffle,
        drop_last = drop_last,
        pin_memory = True
    )
