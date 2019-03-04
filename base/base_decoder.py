import torch

from kws_decoder.ctc_decoder import CTCDecoder
from kws_decoder.kaldi_decoder import KaldiDecoder


def get_decoder(model_path, keywords, tmp_dir):
    assert model_path.endswith(".pth")
    config = torch.load(model_path, map_location='cpu')['config']

    if 'CTC' in config['arch']['loss']['name'] or 'ctc' in config['arch']['loss']['name']:
        return CTCDecoder(model_path, keywords, tmp_dir)
    else:
        return KaldiDecoder(model_path, keywords, tmp_dir)
