from torch.nn import Module
import numpy as np
from nn_.metrics.ctc_utils.error_rate.error_rate import cer
from utils.logger_config import logger
import ctcdecode


class PhnErrorRate(Module):

    def __init__(self, vocabulary_size, batch_ordering):
        super().__init__()
        # WARINIG dont use chr(0)
        vocabulary_size += 1  # TODO unify blank label stuff
        self.vocabulary = [chr(c) for c in list(range(65, 65 + 58)) + list(range(65 + 58 + 69, 65 + 58 + 69 + 500))][
                          :vocabulary_size]
        self.decoder = ctcdecode.CTCBeamDecoder(self.vocabulary, log_probs_input=True, beam_width=1)
        self.batch_ordering = batch_ordering

    @property
    def cpu_only(self):
        return True

    def convert_to_string(self, tokens, vocab, seq_len):
        assert isinstance(tokens, list) or (isinstance(tokens, np.ndarray) and len(tokens.shape) == 1)
        assert isinstance(seq_len, int)
        assert isinstance(vocab, list)
        assert min(tokens[0:seq_len]) >= 0
        # assert max(tokens[0:seq_len]) < len(vocab), max(tokens[0:seq_len])
        if not max(tokens[0:seq_len]) < len(vocab):
            print("here")
        return ''.join([vocab[x] for x in tokens[0:seq_len]])

    def forward(self, output, target):

        if self.batch_ordering == 'NCT':
            # NCT (NCL) -> NTC required for ctcdecode
            logits = output['out_phn'].permute(0, 2, 1)
        elif self.batch_ordering == 'TNCL':
            assert len(output['out_phn'].shape) == 3
            # TNC -> NTC required for ctcdecode
            logits = output['out_phn'].permute(1, 0, 2)
        else:
            raise NotImplementedError

        # decoder expects batch first
        target_sequence_lengths = target['target_sequence_lengths']
        input_sequence_lengths = target['input_sequence_lengths']
        batch_size = len(input_sequence_lengths)
        # batch x seq x label_size

        assert logits.shape[0] == batch_size
        # N x L x C expected
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(logits)

        curr_idx = 0
        all_labels = []
        for b in range(batch_size):
            all_labels.append(target['lab_phn'][curr_idx: curr_idx + target_sequence_lengths[b].item()])
            curr_idx += target_sequence_lengths[b].item()

        per_all = []
        for b in range(batch_size):
            if out_seq_len[b].item() > 0:
                _decoded = self.convert_to_string(beam_result[b][0].cpu().numpy(), self.vocabulary,
                                                  out_seq_len[b].item())
                _labels = all_labels[b].cpu().numpy()
                labels = self.convert_to_string(_labels, self.vocabulary, len(_labels))

                per = cer(_decoded, labels, ignore_case=True, remove_space=True)
                per_all.append(per)
            else:
                pass  # TODO
                # logger.debug("Skip metric since the decoded string is of length 0.")

        if len(per_all) > 0:
            return np.mean(per_all)
        else:
            return -1
