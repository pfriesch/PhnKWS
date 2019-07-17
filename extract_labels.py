from data import kaldi_io


def load_labels(label_folder, label_opts):
    labels_loaded = \
        {k: v for k, v in
         kaldi_io.read_vec_int_ark(
             f'gunzip -c {label_folder}/ali*.gz | {label_opts} {label_folder}/final.mdl ark:- ark:-|')}
    assert len(labels_loaded) > 0
    return labels_loaded


if __name__ == '__main__':
    lab = load_labels("/Volumes/Backup/PhnKwsIndividualProjectBackup/data/libs/kaldi/egs/librispeech/s5/exp/tri4b_ali_test_clean_100/", "kaldi-docker ali-to-pdf")
    print(lab)