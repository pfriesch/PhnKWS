from torch.nn import Module
import ctcdecode
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np

from _nn.metrics.ctc_utils.error_rate.error_rate import cer
from utils.logger_config import logger


class PhnErrorRate(Module):

    def __init__(self, vocabulary_size):
        super().__init__()
        # WARINIG dont use chr(0)
        self.vocabulary = [chr(c) for c in list(range(65, 65 + 58)) + list(range(65 + 58 + 69, 65 + 58 + 69 + 500))][
                          :vocabulary_size]
        self.decoder = ctcdecode.CTCBeamDecoder(self.vocabulary, log_probs_input=True, beam_width=1)

    def convert_to_string(self, tokens, vocab, seq_len):
        assert isinstance(tokens, list) or (isinstance(tokens, np.ndarray) and len(tokens.shape) == 1)
        assert isinstance(seq_len, int)
        assert isinstance(vocab, list)
        return ''.join([vocab[x] for x in tokens[0:seq_len]])

    def forward(self, output, target):
        # decoder expects batch first
        logits = output['out_phn'].permute(1, 0, 2)
        # assert isinstance(target["lab_phn"], PackedSequence)
        batch_size = logits.shape[0]
        max_seq_len = logits.shape[1]

        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(logits, output['sequence_lengths'])

        labels, label_lengths = pad_packed_sequence(target['lab_phn']['label'])

        per_all = []
        for b in range(batch_size):
            if out_seq_len[b].item() > 0:
                _decoded = self.convert_to_string(beam_result[b][0].numpy(), self.vocabulary,
                                                  out_seq_len[b].item())

                _labels = self.convert_to_string(labels[:, b].numpy(), self.vocabulary, label_lengths[b].item())

                per = cer(_decoded, _labels, ignore_case=True, remove_space=True)
                per_all.append(per)
            else:
                logger.debug("Skip metric since the decoded string is of length 0.")

        if len(per_all) > 0:
            return np.mean(per_all)
        else:
            return -1
