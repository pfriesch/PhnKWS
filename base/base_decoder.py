import torch

from kws_decoder.ctc_decoder import CTCDecoder
from kws_decoder.kaldi_decoder import KaldiDecoder


def get_decoder(model_path, keywords, tmp_dir):
    assert model_path.endswith(".pth")
    config = torch.load(model_path, map_location='cpu')['config']

    if not config['arch']['framewise_labels']:
        return CTCDecoder(model_path, keywords, tmp_dir)
    else:
        return KaldiDecoder(model_path, keywords, tmp_dir)
