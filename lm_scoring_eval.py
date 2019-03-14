import json
import os
import tempfile

import torch

from base.utils import resume_checkpoint
from kws_decoder.eesen_decoder_lexicon.prepare_decode_graph import make_ctc_decoding_graph
from nn_.networks.LSTM_cd_Net import LSTM_cd_Net
from nn_.registries.metrics_registry import metrics_init
from nn_.registries.model_registry import model_init
from trainer.eval import evaluate
from utils.logger_config import logger
from utils.util import ensure_dir
from utils.utils import check_environment

if __name__ == '__main__':
    check_environment()

    # final_architecture1 = torch.load(
    #     "/mnt/data/pytorch-kaldi_cfg/exp/libri_LSTM_fbank/exp_files/final_architecture1.pkl", map_location='cpu')
    # final_architecture2 = torch.load(
    #     "/mnt/data/pytorch-kaldi_cfg/exp/libri_LSTM_fbank/exp_files/final_architecture2.pkl", map_location='cpu')

    mdl = LSTM_cd_Net(input_feat_name='fbank', input_feat_length=40, lab_cd_num=3480)
    mdl.load_cfg()

    tmp_root_dir = '/mnt/data/tmp_kws_eval'
    if not os.path.exists(tmp_root_dir):
        os.makedirs(tmp_root_dir)
    # tmp_dir = tempfile.TemporaryDirectory(dir=tmp_root_dir)
    # _tmp_dir = tmp_dir.name
    _tmp_dir = f"{tmp_root_dir}/ctc_decoing_1gram_lm"

    # model_path = "/mnt/data/pytorch-kaldi/exp_finished_runs_backup/libri_MLP_fbank_ctc_20190308_182124/checkpoints/checkpoint_e5.pth"

    # assert model_path.endswith(".pth")
    # config = torch.load(model_path, map_location='cpu')['config']

    # TODO remove
    # config['exp']['save_dir'] = "/mnt/data/pytorch-kaldi/exp_TIMIT_MLP_FBANK"

    # model = model_init(config)
    model = mdl
    metrics = metrics_init(config, model)
    # TODO GPU decoding

    max_seq_length_train_curr = -1

    out_dir = os.path.join(config['exp']['save_dir'], config['exp']['name'])

    # setup directory for checkpoint saving
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')

    # Save configuration file into checkpoint directory:
    ensure_dir(checkpoint_dir)
    config_save_path = os.path.join(out_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4, sort_keys=False)

    epoch, global_step, _ = resume_checkpoint(model_path, model, logger)

    phoneme_dict = config['dataset']['dataset_definition']['phoneme_dict']

    graph_dir = "/mnt/data/tmp_kws_eval/ctc_decoing_1gram_lm/tmp/graph_dir"
    if not os.path.exists(graph_dir):
        arpa_lm_path = "/mnt/data/libs/kaldi/egs/librispeech/s5/data/local/lm/3-gram.pruned.3e-7.arpa.gz"

        graph_dir = make_ctc_decoding_graph(arpa_lm_path, phoneme_dict.phoneme2reducedIdx, _tmp_dir,
                                            draw_G_L_fsts=False)
        print(graph_dir)
        graph_path = os.path.join(graph_dir, "TLG.fst")
        assert os.path.exists(graph_path)
        words_path = os.path.join(graph_dir, "words.txt")
        # alignment_model_path = os.path.join(graph_dir, "final.mdl")
        # assert os.path.exists(alignment_model_path) checkpoint['dataset_sampler_state'] is none

    test_data = config['dataset']['data_use']['test_with']
    evaluate(model,
             metrics,
             device='cpu',
             out_folder=os.path.join(config['exp']['save_dir'], config['exp']['name']),
             exp_name=config['exp']['name'],
             max_label_length=-1,
             epoch=epoch,
             dataset_type=config['training']['dataset_type'],
             data_cache_root=config['exp']['data_cache_root'],
             test_with=test_data,
             all_feats_dict=config['dataset']['dataset_definition']['datasets'][test_data]['features'],
             features_use=config['dataset']['features_use'],
             all_labs_dict=config['dataset']['dataset_definition']['datasets'][test_data]['labels'],
             labels_use=config['dataset']['labels_use'],
             phoneme_dict=config['dataset']['dataset_definition']['phoneme_dict'],
             decoding_info=config['dataset']['dataset_definition']['decoding'],
             lab_graph_dir=graph_dir)

    # tmp_dir.cleanup()
