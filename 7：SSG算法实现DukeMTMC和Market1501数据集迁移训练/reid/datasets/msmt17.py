from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import read_json
from ..utils.serialization import write_json

class MSMT17(object):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html
    
    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'msmt17'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'MSMT17_V1/train')
        self.test_dir = osp.join(self.dataset_dir, 'MSMT17_V1/test')
        self.list_train_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_gallery.txt')

        self._check_before_run()
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, self.list_train_path)
        val, num_val_pids, num_val_imgs = self._process_dir(self.train_dir, self.list_val_path)
        query, num_query_pids, num_query_imgs = self._process_dir(self.test_dir, self.list_query_path)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.test_dir, self.list_gallery_path)

        #train += val
        #num_train_imgs += num_val_imgs

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> MSMT17 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.trainval = train
        self.val = val
        self.query = query
        self.gallery = gallery

        self.num_train_ids = num_train_pids
        self.num_trainval_ids = num_train_pids
        self.num_val_ids = num_val_pids
        self.num_query_ids = num_query_pids
        self.num_gallery_ids = num_gallery_pids
        self.images_dir = ''
        self.num_cams = 15

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2]) - 1
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)
        num_imgs = len(dataset)
        num_pids = len(pid_container)
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset, num_pids, num_imgs
