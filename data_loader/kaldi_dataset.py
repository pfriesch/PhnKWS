import time

import torch

from data_loader.data_util import read_lab_fea, apply_context, make_big_chunk


class KaldiDataset(object):

    def __init__(self, fea_dict, lab_dict, context_left, context_right, tensorboard_logger, debug=False):
        self.tensorboard_logger = tensorboard_logger
        start_time = time.time()

        feat_dict, lab_dict = read_lab_fea(fea_dict, lab_dict)

        if debug:
            for lab in lab_dict:
                lab_dict[lab] = dict(list(lab_dict[lab].items())[:30])
            for fea in feat_dict:
                feat_dict[fea] = dict(list(feat_dict[fea].items())[:30])

        # TODO split files that are too long
        lab_dict = apply_context(lab_dict, context_left, context_right)

        # TODO make multiple chunks if too big
        sample_name, feat_chunks, lab_chunks = make_big_chunk(feat_dict, lab_dict)
        self.feat_chunks = {fea: torch.from_numpy(feat_chunks[fea]).float()
                            for fea, v in feat_chunks.items()}
        self.lab_chunks = {lab: torch.from_numpy(lab_chunks[lab]).long()
                           for lab, v in lab_chunks.items()}
        self.sample_names, self.samples = zip(*list(sample_name.items()))

        self.fea_dim = {fea: v.shape[1] for fea, v in self.feat_chunks.items()}

        elapsed_time_load = time.time() - start_time
        self.tensorboard_logger.add_scalar("init_dataset", elapsed_time_load)

    def move_to(self, device):
        start_time = time.time()

        # Called "move to" to indicated difference to pyTorch .to(). This function mutates this object.
        self.feat_chunks = {k: v.to(device) for k, v in self.feat_chunks.items()}
        self.lab_chunks = {k: v.to(device) for k, v in self.lab_chunks.items()}

        elapsed_time_load = time.time() - start_time
        self.tensorboard_logger.add_scalar("init_dataset", elapsed_time_load)

    def __getitem__(self, index):
        return (self.sample_names[index],
                {fea: self.feat_chunks[fea][v['start_idx']:v['end_idx']]
                 for fea, v in self.samples[index]['fea'].items()},
                {lab: self.lab_chunks[lab][v['start_idx']:v['end_idx']]
                 for lab, v in self.samples[index]['lab'].items()})

    def __len__(self):
        return len(self.samples)
