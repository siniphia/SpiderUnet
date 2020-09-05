import os, glob, random
import tensorflow as tf
import numpy as np
import pandas as pd

# modules for medical data reading
import pydicom as dicom
import nibabel as nib
import scipy.io as io
from tensorflow import keras


def get_dataset(dataset_type, dataset_category, dataset_size, seq_len, seq_interval, label_type='mask', do_augmentation=True):
    """
    Description
        The highest-level dataset getter function

    Params
        dataset_type (str) : 'train', 'val', 'test'
        dataset_category (str) : 'clinical_mra', 'synthetic_mra', 'hepatic_vessel', 'lv_vessel'
        dataset_size (int) : decide dataset size to return
        seq_len (int) : the sampling size of an image sequence
        seq_interval (int) : the sampling interval
        label_type (str, optional) : 'mask', 'bifur', 'centerline', 'points', 'radius'
        do_augmentation (bool, optional) : apply augmentation strategies (e.g. std, resize, crop)

    Returns
        dataset (tf.data.Dataset) : paired dataset object of image and label (shape = [t, h, w, c])
    """
    # 1 - Create Raw Dataset
    if dataset_category == 'synthetic_mra':
        img, lbl, seq = _get_synthetic_mra_path(dataset_type, dataset_size, seq_len, seq_interval, label_type)
        dataset = tf.data.Dataset.from_tensor_slices((img, lbl, seq))
        dataset = dataset.map(lambda x, y, z: __py_synthetic_mra(x, y, z, seq_len))
    elif dataset_category == 'clinical_mra':
        img, lbl, seq = _get_clinical_mra_path(dataset_type, dataset_size, seq_len, seq_interval, label_type)
        dataset = tf.data.Dataset.from_tensor_slices((img, lbl, seq))
        dataset = dataset.map(lambda x, y, z: __py_clinical_mra(x, y, z, seq_len))
    elif dataset_category == 'hepatic_vessel':
        img, lbl, seq = _get_hepatic_vessel_path(dataset_type, dataset_size, seq_len, seq_interval, label_type)
        dataset = tf.data.Dataset.from_tensor_slices((img, lbl, seq))
        dataset = dataset.map(lambda x, y, z: __py_hepatic_vessel(x, y, z, seq_len))
    elif dataset_category == 'lv_vessel':
        img, lbl, seq, phase = _get_lv_vessel_path(dataset_type, dataset_size, seq_len, seq_interval, label_type)
        dataset = tf.data.Dataset.from_tensor_slices((img, lbl, seq, phase))
        dataset = dataset.map(lambda x, y, z, w: __py_lv_vessel(x, y, z, w, seq_len))
    else:
        print('Invalid dataset category :', dataset_category)
        return

    # dataset = tf.data.Dataset.from_tensor_slices((img, lbl, seq))
    # dataset = dataset.map(lambda x, y, z: py_function_wrapper(x, y, z, seq_len, dataset_category))
    dataset = dataset.map(lambda x, y: image_augmentation(x, y, do_augmentation))

    return dataset


def _get_synthetic_mra_path(dataset_type, dataset_size, seq_len, seq_interval, label_type='mask'):
    """
    Description
        - Artificial vessel dataset for pre-training purpose
        - https://github.com/giesekow/deepvesselnet/wiki/Datasets
    """

    # 1 - get file paths
    root_dir, work_dir = 'D:\\mra\\dataset\\', 'synthetic'
    img_path = sorted(glob.glob(os.path.join(root_dir, 'raw', '*.nii.gz')))
    lbl_path = sorted(glob.glob(os.path.join(root_dir, label_type, '*.nii.gz')))

    # 2 - slice dataset by type
    if dataset_type == 'train':
        img_path, lbl_path = img_path[:126], lbl_path[:126]
    elif dataset_type == 'val':
        img_path, lbl_path = img_path[126:131], lbl_path[126:131]
    else:
        img_path, lbl_path = img_path[131:], lbl_path[131:]

    # 3 - declare return lists
    img_list, lbl_list, seq_list = [], [], []
    seq_idx = 0

    # 4 - prepare slicing z-axis of data with fixed interval
    for i in range(len(img_path)):
        max_idx = nib.load(img_path[i]).shape[-1]
        while seq_idx + seq_len - 1 < max_idx and dataset_size > len(img_list):
            img_list += [img_path[i]]
            lbl_list += [lbl_path[i]]
            seq_list += [seq_idx]
            seq_idx += seq_interval
        seq_idx = 0

    # 5 - returns
    print('> Loaded Synthetic %s Dataset\n\t- Number of Image Sequences : %d\n\t- Label Type : %s' % (dataset_type.title(), len(img_list), label_type))
    return img_list, lbl_list, seq_list


def __get_synthetic_mra_data(img, lbl, seq, seq_len):
    img = nib.load(img.numpy().decode('utf-8')).dataobj[..., seq:seq + seq_len]  # h, w, t
    img = np.expand_dims(img, axis=-1)  # h, w, t, c
    img = np.transpose(img, [2, 0, 1, 3])  # t, h, w, c
    img = img.astype(np.float32)

    lbl = nib.load(lbl.numpy().decode('utf-8')).dataobj[..., seq:seq + seq_len]  # h, w, t
    lbl = np.expand_dims(lbl, axis=-1)  # h, w, t, c
    lbl = np.transpose(lbl, [2, 0, 1, 3])  # t, h, w, c
    lbl = lbl.astype(np.uint8)

    return img, lbl


def _get_clinical_mra_path(dataset_type, dataset_size, seq_len, seq_interval, label_type='mask'):
    """
    Description
        - Internal TOF MRA images from Seoul National University Bundang Hospital
    """
    # 1 - get dataframe
    root_dir = 'D:\\mra\\dataset\\'
    df = pd.read_excel(os.path.join(root_dir, 'mra_db.xlsx'))[['Code', 'Type', 'PatientNum', 'FilePath', 'MaskPath', 'Label']]

    # 2 - declare return lists
    img_list, lbl_list, seq_list = [], [], []

    # 3 - prepare slicing z-axis of data with fixed interval
    for idx, row in df.iterrows():
        # only consider correponding run type
        if row['Type'] == dataset_type:
            raw_arr = nib.load(str(row['FilePath'])).get_fdata()
            current_idx, start_idx, end_idx = 0, 0, raw_arr.shape[-1] - 1

            # find start and end index
            for j in range(end_idx + 1):
                if np.max(raw_arr[:, :, j]) != 0:
                    start_idx = j
                    break
            for j in range(end_idx + 1):
                if np.max(raw_arr[:, :, end_idx - j]) != 0:
                    end_idx = end_idx - j
                    break

            current_idx = start_idx
            while current_idx + seq_len - 1 < end_idx and len(img_list) < dataset_size:
                img_list += [row['FilePath']]
                seq_list += [current_idx]
                if label_type == 'mask':
                    lbl_list += [str(row['MaskPath'])]
                else:
                    lbl_list += [int(row['Label'])]
                current_idx += seq_interval

    # 4 - returns
    print('> Loaded Clinical %s Dataset\n\t- total number of image seqences : %d\n\t- the number of slices in an image sequence : %d\n\t- label type : %s'
          % (dataset_type.title(), len(img_list), seq_len, label_type))
    return img_list, lbl_list, seq_list


def __get_clinical_mra_data(img, lbl, seq, seq_len):
    # convert tensor args into pythonic objects
    img = img.numpy().decode('utf-8')
    lbl = lbl.numpy().decode('utf-8')
    seq = int(seq.numpy())
    seq_len = int(seq_len.numpy())

    # image
    img = nib.load(img).dataobj
    w, h = np.shape(img)[0], np.shape(img)[1]
    img = img[w//2-128:w//2+128, h//2-128:h//2+128, seq:seq+seq_len]  # 256, 256, t
    # img = nib.load(img).dataobj[..., seq:seq + seq_len]  # h, w, t
    img = np.expand_dims(img, axis=-1)  # h, w, t, c
    img = np.transpose(img, [2, 1, 0, 3])  # t, h, w, c
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # label
    lbl = nib.load(lbl).dataobj
    w, h = np.shape(lbl)[0], np.shape(lbl)[1]
    lbl = lbl[w//2-128:w//2+128, h//2-128:h//2+128, seq:seq+seq_len]
    # lbl = nib.load(lbl).dataobj[..., seq:seq + seq_len]  # h, w, t
    lbl = lbl.astype(np.uint8)
    lbl = np.expand_dims(lbl, axis=-1)  # h, w, t, c
    lbl = np.transpose(lbl, [2, 1, 0, 3])  # t, h, w, c

    return img, lbl


def _get_hepatic_vessel_path(dataset_type, dataset_size, seq_len, seq_interval, label_type='mask'):
    """
    Description
        - Hepatic vessel and tumor dataset from Decathlon Challenge
        - http://medicaldecathlon.com/
    """
    # 1 - get file paths
    root_dir = 'D:\\2_hepatic_vessel'
    img_path = sorted(glob.glob(os.path.join(root_dir, 'Task08_HepaticVessel', 'imagesTr', 'hepaticvessel_*.nii.gz')))
    lbl_path = sorted(glob.glob(os.path.join(root_dir, 'Task08_HepaticVessel', 'labelsTr', 'hepaticvessel_*.nii.gz')))

    # 2 - slice dataset by type (total 303)
    if dataset_type == 'train':
        img_path, lbl_path = img_path[:253], lbl_path[:253]
    elif dataset_type == 'val':
        img_path, lbl_path = img_path[253:278], lbl_path[253:278]
    else:
        img_path, lbl_path = img_path[278:302], lbl_path[278:302]

    # 3 - declare return lists
    img_list, lbl_list, seq_list = [], [], []

    # 4 - prepare slicing z-axis of data with fixed interval
    for i in range(len(lbl_path)):
        raw_arr = nib.load(lbl_path[i]).get_fdata()
        raw_arr = np.where(raw_arr == 2, 0, raw_arr)
        current_idx, start_idx, end_idx = 0, 0, raw_arr.shape[-1] - 1

        # set start and end index within label-existing slices
        if dataset_type == 'train':
            for j in range(end_idx + 1):
                if np.max(raw_arr[:, :, j]) != 0:
                    start_idx = j
                    break
            for j in range(end_idx + 1):
                if np.max(raw_arr[:, :, end_idx - j]) != 0:
                    end_idx = end_idx - j
                    break

        # print('from %d to %d...' % (start_idx, end_idx))
        current_idx = start_idx
        while current_idx + seq_len - 1 < end_idx and len(img_list) < dataset_size:
            img_list += [img_path[i]]
            lbl_list += [lbl_path[i]]
            seq_list += [current_idx]
            current_idx += seq_interval

    # 5 - returns
    print('> Loaded Hepatic Vessel %s Dataset\n\t- total number of image seqences : %d\n\t- the number of slices in an image sequence : %d\n\t- label type : %s'
          % (dataset_type.title(), len(img_list), seq_len, label_type))

    return img_list, lbl_list, seq_list


def __get_hepatic_vessel_data(img, lbl, seq, seq_len):
    # convert tensors into pythonic objects
    img = img.numpy().decode('utf-8')
    lbl = lbl.numpy().decode('utf-8')
    seq = int(seq.numpy())
    seq_len = int(seq_len.numpy())

    raw = nib.load(img).dataobj
    raw = np.asarray(raw).astype(np.float32)

    # clip and normalize
    max_threshold, min_threshold = 300., 0.
    raw = np.clip(raw, min_threshold, max_threshold)
    raw = (raw - min_threshold) / (max_threshold - min_threshold)

    # image
    img = raw[..., seq:seq + seq_len]  # h, w, t
    img = np.transpose(img, [2, 1, 0])  # t, h, w
    img = img[:, 128:384, 192:448]
    img = np.expand_dims(img, axis=-1)  # t, h, w, c

    # label
    lbl = nib.load(lbl).dataobj[..., seq:seq + seq_len]  # h, w, t
    lbl = lbl.astype(np.uint8)
    lbl = np.transpose(lbl, [2, 1, 0])  # t, h, w
    lbl = np.where(lbl == 2, 0, lbl)
    lbl = lbl[:, 128:384, 192:448]
    lbl = np.expand_dims(lbl, axis=-1)  # h, w, t, c

    return img, lbl


def _get_lv_vessel_path(dataset_type, dataset_size, seq_len, seq_interval, label_type='mask'):
    """
    Description
        - LV endo/epicardiac MRI dataset from York Univ.
        - http://www.cse.yorku.ca/~mridataset/
    """
    # 1 - get file paths
    root_dir = 'D:\\3_lv_mri'
    img_path = sorted(glob.glob(os.path.join(root_dir, 'images', 'sol_yxzt_pat*.mat')))
    lbl_path = sorted(glob.glob(os.path.join(root_dir, 'labels', 'manual_seg_32points_pat*.mat')))
    # 2 - slice dataset by type (total 33)
    if dataset_type == 'train':
        img_path, lbl_path = img_path[:27], lbl_path[:27]
    elif dataset_type == 'val':
        img_path, lbl_path = img_path[27:30], lbl_path[27:30]
    else:
        img_path, lbl_path = img_path[30:33], lbl_path[30:33]

    # 3 - declare return lists
    img_list, lbl_list, seq_list, phase_list = [], [], [], []

    # 4 - prepare slicing z-axis of data with fixed interval
    for i in range(len(lbl_path)):
        raw_arr = io.loadmat(lbl_path[i])['manual_seg_32points']  # (n, 20) array, each element is (65, 2) or (1, 1)
        slices, phases = np.shape(raw_arr)[0], np.shape(raw_arr)[1]  # n, 20

        for phase_num in range(phases):
            current_idx, start_idx, end_idx = 0, 0, slices - 1

            # find start and end index from center slice
            center_idx = int(end_idx / 2)
            while True:
                prev_shape = np.shape(raw_arr[center_idx - 1][phase_num])
                curr_shape = np.shape(raw_arr[center_idx][phase_num])
                next_shape = np.shape(raw_arr[center_idx + 1][phase_num])

                if prev_shape == (65, 2) and curr_shape == (65, 2):
                    center_idx -= 1
                elif prev_shape == (1, 1) and curr_shape == (1, 1):
                    center_idx += 1
                elif prev_shape == (1, 1) and curr_shape == (65, 2) and next_shape == (65, 2):
                    start_idx = center_idx
                    break

            center_idx = int(end_idx / 2)
            while center_idx < end_idx:
                prev_shape = np.shape(raw_arr[center_idx - 1][phase_num])
                curr_shape = np.shape(raw_arr[center_idx][phase_num])
                next_shape = np.shape(raw_arr[center_idx + 1][phase_num])

                if curr_shape == (65, 2) and next_shape == (65, 2):
                    center_idx += 1
                    end_idx = center_idx
                elif curr_shape == (1, 1) and next_shape == (1, 1):
                    center_idx -= 1
                    end_idx = center_idx
                elif prev_shape == (65, 2) and curr_shape == (65, 2) and next_shape == (1, 1):
                    end_idx = center_idx
                    break

            # for k in range(center_idx):
            #     if np.shape(raw_arr[center_idx + k][phase_num]) != (65, 2):
            #         end_idx = center_idx + k - 1
            #         break

            # print('[Seq %02d] phase %d, start index : %d, end index : %d' % (i, phase_num, start_idx, end_idx))
            current_idx = start_idx
            while current_idx + seq_len - 1 <= end_idx and len(img_list) < dataset_size:
                print('[File %02d] phase %d, start %d, end %d, current index : %d, seq len : %d, seq shape :' % (i, phase_num, start_idx, end_idx, current_idx, seq_len), np.shape(raw_arr[current_idx][phase_num]))
                # print('[File %02d] phase %d, seq shape : ' % (i, phase_num), np.shape(raw_arr[current_idx][phase_num]))
                img_list += [img_path[i]]
                lbl_list += [lbl_path[i]]
                seq_list += [current_idx]
                phase_list += [phase_num]
                current_idx += seq_interval

    # 5 - returns
    print('> Loaded LV Vessel %s Dataset\n\t- total number of image seqences : %d\n\t- the number of slices in an image sequence : %d\n\t- label type : %s'
          % (dataset_type.title(), len(img_list), seq_len, label_type))
    return img_list, lbl_list, seq_list, phase_list


def __get_lv_vessel_data(img, lbl, seq, phase, seq_len):
    def __check(p1, p2, base_array):
        # https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array
        idxs = np.indices(base_array.shape)  # Create 3D array of indices
        p1 = p1.astype(float)
        p2 = p2.astype(float)

        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = np.sign(p2[0] - p1[0])
        return idxs[1] * sign <= max_col_idx * sign

    def __to_mask(vertices, shape):
        # https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array
        base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros
        fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

        # Create check array for each edge segment, combine into fill array
        for k in range(vertices.shape[0]):
            fill = np.all([fill, __check(vertices[k - 1], vertices[k], base_array)], axis=0)
        base_array[fill] = 1

        return np.swapaxes(base_array, 0, 1)

    # convert tensor args into pythonic objects
    img = img.numpy().decode('utf-8')
    lbl = lbl.numpy().decode('utf-8')
    seq = int(seq.numpy())
    phase = int(phase.numpy())
    seq_len = int(seq_len.numpy())

    # read array parts of mat data
    raw_img = io.loadmat(img)['sol_yxzt']  # h, w, x, y
    raw_lbl = io.loadmat(lbl)['manual_seg_32points']  # x, y

    img_dims, lbl_dims = np.shape(raw_img), np.shape(raw_lbl)

    # flat_img = np.reshape(raw_img, (img_dims[0], img_dims[1], img_dims[2] * img_dims[3]))  # h, w, x * y
    # flat_lbl = np.reshape(raw_lbl, (lbl_dims[-2] * lbl_dims[-1]))  # x * y

    # image; normalize from 0 to 1
    img = raw_img[:, :, seq:seq + seq_len, phase]
    # img = flat_img[..., seq:seq + seq_len]  # h, w, t
    img = np.expand_dims(img, axis=-1)  # h, w, t, c
    img = np.transpose(img, [2, 0, 1, 3])  # t, h, w, c
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # label
    lbl_list = []
    for i in range(seq, seq + seq_len):
        epic = __to_mask(raw_lbl[i][phase][0:32], [img_dims[0], img_dims[1]])
        endo = __to_mask(raw_lbl[i][phase][33:65], [img_dims[0], img_dims[1]])
        # epic = __to_mask(flat_lbl[i][0:32], [img_dims[0], img_dims[1]])
        # endo = __to_mask(flat_lbl[i][33:65], [img_dims[0], img_dims[1]])
        lbl_list += [epic - endo]  # list of (h, w)
    lbl = np.stack(lbl_list, axis=-1)  # h, w, t
    lbl = np.expand_dims(lbl, axis=-1)  # h, w, t, c
    lbl = np.transpose(lbl, [2, 0, 1, 3])  # t, h, w, c
    lbl = lbl.astype(np.uint8)

    return img, lbl


def __py_synthetic_mra(img, lbl, seq, seq_len):
    image, label = tf.py_function(__get_synthetic_mra_data, [img, lbl, seq, seq_len], [tf.float32, tf.uint8])
    image.set_shape([seq_len, None, None, 1])
    label.set_shape([seq_len, None, None, 1])

    return image, label


def __py_clinical_mra(img, lbl, seq, seq_len):
    image, label = tf.py_function(__get_clinical_mra_data, [img, lbl, seq, seq_len], [tf.float32, tf.uint8])
    image.set_shape([seq_len, None, None, 1])
    label.set_shape([seq_len, None, None, 1])

    return image, label


def __py_hepatic_vessel(img, lbl, seq, seq_len):
    image, label = tf.py_function(__get_hepatic_vessel_data, [img, lbl, seq, seq_len], [tf.float32, tf.uint8])
    image.set_shape([seq_len, None, None, 1])
    label.set_shape([seq_len, None, None, 1])

    return image, label


def __py_lv_vessel(img, lbl, seq, phase, seq_len):
    image, label = tf.py_function(__get_lv_vessel_data, [img, lbl, seq, phase, seq_len], [tf.float32, tf.uint8])
    image.set_shape([seq_len, None, None, 1])
    label.set_shape([seq_len, None, None, 1])

    return image, label


def image_augmentation(image, label, do_augmentation, to_onehot=True, crop_ratio=.75, rsz_w=256, rsz_h=256, c_min=1., c_max=2, b_val=.15):
    """
    Description
        Transform images of created tf.data.Dataset
        - image : standardization -> resize -> translate -> contrast -> brightness
        - label : resize -> translate

    Params
        image : (t, h, w, c)
        label : (t, h, w, c)

    Returns
        Augmented image and label objects with same shape of inputs
    """
    # trsl_px = tf.random.uniform([2], -25, 25, dtype=tf.int32)
    # rot_angle = tf.random_uniform([1], -10, 10, dtype=tf.int32)

    ori_images = tf.unstack(image, axis=0)  # (t, h, w, c)
    ori_labels = tf.unstack(label, axis=0)  # (t, h, w, c)
    crop_ratio = random.uniform(crop_ratio, 1.)
    aug_images = []
    aug_labels = []

    # image part
    for img in ori_images:  # h, w, c
        img = tf.image.per_image_standardization(img)
        if do_augmentation:
            img = tf.image.central_crop(img, crop_ratio)
            img = tf.image.resize(img, size=[rsz_h, rsz_w])
            img = tf.image.random_contrast(img, c_min, c_max)
            img = tf.image.random_brightness(img, b_val)
        else:
            img = tf.image.resize(img, size=[rsz_h, rsz_w])
        aug_images += [img]

    image = tf.stack(aug_images, axis=0)  # t, h, w, c
    if image.get_shape()[0] == 1:
        image = tf.squeeze(image, axis=0)  # h, w, c  ; delete time axis when t=1

    # mask part
    for lbl in ori_labels:
        if do_augmentation:
            lbl = tf.image.central_crop(lbl, crop_ratio)
        lbl = tf.image.resize(lbl, size=[rsz_h, rsz_w])  # h, w, c
        lbl = tf.where(lbl < 0.5, 0, 1)
        aug_labels += [lbl]
    label = tf.stack(aug_labels, axis=0)  # t, h, w, c
    label = tf.squeeze(label, axis=-1)  # t, h, w ; for sparse loss

    if to_onehot:
        label = tf.one_hot(label, depth=2)  # t, h, w, class_num ; for dense loss
    if label.get_shape()[0] == 1:
        label = tf.squeeze(label, axis=0)  # h, w, class_num ; delete time axis when t=1

    return image, label


# img, lbl, seq, phase = _get_lv_vessel_path('train', 10000, 5, 3)
# for i in range(10):
#     print(lbl[i], seq[i], phase[i])
