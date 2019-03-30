import torch

import torch.nn.functional as F
from torch import nn

from data import PADDING_IGNORE_INDEX


class CELoss(nn.Module):

    def __init__(self, batch_ordering, weight_mono=1.0):
        super().__init__()
        self.weight_mono = weight_mono
        self.batch_ordering = batch_ordering

    def forward(self, output, target):

        if self.weight_mono != 1:
            raise NotImplementedError

        # batch_size = output['out_cd'].shape[0]
        # seq_len = output['out_cd'].shape[2]

        loss_dict = {}
        for _output, _target in zip(output, target):

            if len(target[_target].shape) == 2:

                assert _output.split("_")[1] == _target.split("_")[1], \
                    (_output.split("_")[1], _target.split("_")[1])
                _name = _output.split("_")[1]

                _num_out = output[_output].shape[1]
                _target_max = target[_target].view(-1).max()
                assert _target_max < _num_out, \
                    f"got max {_target_max}," \
                    + f" expeced {_num_out} (min: {target[_target].view(-1).min()}) for {_output} and {_target}"
                _loss = F.nll_loss(output[_output],
                                   target[_target],
                                   ignore_index=PADDING_IGNORE_INDEX)
                loss_dict["loss_" + _name] = _loss
            elif len(target[_target].shape) == 1 \
                    and 'target_sequence_lengths' in target \
                    and 'input_sequence_lengths' in target:
                max_seq_len, batch, n_outs = output[_output].shape
                if batch != 1:
                    raise NotImplementedError
                else:
                    _name = _output.split("_")[1]
                    _loss = F.nll_loss(output[_output].squeeze(1),
                                       target[_target].to(dtype=torch.long),
                                       ignore_index=PADDING_IGNORE_INDEX)
                    loss_dict["loss_" + _name] = _loss

        losses = list(loss_dict.keys())
        loss_dict["loss_final"] = 0
        for loss in losses:
            loss_dict["loss_final"] += loss_dict[loss]

        return loss_dict
        # num_cd = output['out_cd'].shape[1]
        # cd_max = target['lab_cd'].view(-1).max()
        # assert cd_max < num_cd, f"got max {cd_max}, expeced {num_cd} (min: {target['lab_cd'].view(-1).min()})"

        # num_mono = output['out_mono'].shape[1]
        # mono_max = target['lab_mono'].view(-1).max()
        # assert mono_max < num_mono, f"got max {mono_max}, expeced {num_mono} (min: {target['lab_mono'].view(-1).min()})"
        #
        # if 'lab_phnframe' in target:
        #     num_phnframe = output['out_phnframe'].shape[1]
        #     phnframe_max = target['lab_phnframe'].view(-1).max()
        #     assert phnframe_max < num_phnframe, \
        #         f"got max {phnframe_max}," \
        #         + f" expeced {num_phnframe} (min: {target['lab_phnframe'].view(-1).min()})"

        # loss_cd = F.nll_loss(output['out_cd'],
        #                      target['lab_cd'],
        #                      ignore_index=PADDING_IGNORE_INDEX)
        # loss_mono = F.nll_loss(output['out_mono'],
        #                        target['lab_mono'],
        #                        ignore_index=PADDING_IGNORE_INDEX)

        # if 'lab_phnframe' in target:
        #     loss_phnframe = F.nll_loss(output['out_phnframe'],
        #                                target['lab_phnframe'],
        #                                ignore_index=PADDING_IGNORE_INDEX)
        #
        #     loss_final = (self.weight_mono * loss_mono) + loss_cd + loss_phnframe
        #     return {"loss_final": loss_final, "loss_cd": loss_cd, "loss_mono": loss_mono,
        #             "loss_phnframe": loss_phnframe}
        #
        # else:
        #
        #     loss_final = (self.weight_mono * loss_mono) + loss_cd
        #     return {"loss_final": loss_final, "loss_cd": loss_cd, "loss_mono": loss_mono}
