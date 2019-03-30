import torch

from kws_decoder.ctc_decoder import CTCDecoder
from kws_decoder.kaldi_decoder import KaldiDecoder
# from kws_decoder.kaldi_decoder_phn import KaldiDecoderPhn


def get_decoder(model_path, keywords, tmp_dir):
    assert model_path.endswith(".pth")
    config = torch.load(model_path, map_location='cpu')['config']

    if 'CTC' in config['arch']['loss']['name'] or 'ctc' in config['arch']['loss']['name']:
        return CTCDecoder(model_path, keywords, tmp_dir)
    else:
        if config['dataset']['labels_use'][0] == "lab_cd":
            return KaldiDecoder(model_path, keywords, tmp_dir)
        # elif config['dataset']['labels_use'][0] == "lab_phnframe":
        #     return KaldiDecoderPhn(model_path, keywords, tmp_dir)
        else:
            raise NotImplementedError
