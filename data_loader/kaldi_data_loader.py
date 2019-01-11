import time

from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader


def collate_fn(sample_list):
    fea_keys = list(sample_list[0][1].keys())
    lab_keys = list(sample_list[0][2].keys())

    sample_names = []
    fea_dict = {k: list() for k in fea_keys}
    lab_dict = {k: list() for k in lab_keys}
    for sample in sample_list:
        sample_names.append(sample[0])
        for fea in fea_dict:
            fea_dict[fea].append(sample[1][fea])
        for lab in lab_dict:
            lab_dict[lab].append(sample[2][lab])

    for fea in fea_dict:
        fea_dict[fea] = pack_sequence(sorted(fea_dict[fea], key=lambda x: x.shape[0], reverse=True))

    for lab in lab_dict:
        lab_dict[lab] = pack_sequence(sorted(lab_dict[lab], key=lambda x: x.shape[0], reverse=True))

    return sample_names, fea_dict, lab_dict


class KaldiDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, shuffle, use_gpu, prefetch_to_gpu,
                 device, num_workers):
        self.dataset = dataset
        self.n_samples = len(self.dataset)
        if prefetch_to_gpu:
            self.dataset.move_to(device)

        # pin_memory = use_gpu and not prefetch_to_gpu
        # TODO packed sequence from collate_fn does not work with pin_memory
        pin_memory = False

        assert (prefetch_to_gpu and num_workers == 0) or not prefetch_to_gpu

        super(KaldiDataLoader, self).__init__(self.dataset,
                                              batch_size,
                                              shuffle,
                                              collate_fn=collate_fn,
                                              pin_memory=pin_memory,
                                              num_workers=num_workers,
                                              drop_last=False)
