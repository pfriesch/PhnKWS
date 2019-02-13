import os
import shutil

from kws_decoder.arpa_utils import make_kw_arpa, UNK_WORD
from utils.logger_config import logger
from utils.utils import run_shell

KALDI_DIR = "/mnt/data/libs/kaldi"


def make_kaldi_decoding_graph(keywords=['alexa', "left", "visual", "marvin"]):
    assert os.getcwd().endswith("kalid_decoder")  # TODO handle the relative folders

    train_graph_dir = f"{KALDI_DIR}/egs/librispeech/s5/exp/tri4b"
    train_dict_folder = f"{KALDI_DIR}/egs/librispeech/s5/data/local/dict_nosp"

    tmpdir = "tmp"
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)
    final_lang_dir = "lang"
    if not hasattr(logger, "logger"):
        logger.configure_logger(tmpdir)

    # TODO
    in_dir = "in_tmp"
    for static_file in ["extra_questions.txt", "nonsilence_phones.txt", "optional_silence.txt", "silence_phones.txt"]:
        if not os.path.exists(f"{in_dir}/{static_file}"):
            shutil.copy(f"{train_dict_folder}/{static_file}", f"{in_dir}/{static_file}")

    libri_lexicon = f"{in_dir}/librispeech-lexicon.txt"
    if not os.path.exists(libri_lexicon):
        run_shell(f"wget -P {in_dir}/ http://www.openslr.org/resources/11/librispeech-lexicon.txt")

    keywords = [kw.upper() for kw in keywords]

    keyword_pron = {}
    lines = ["!SIL SIL\n",
             "<SPOKEN_NOISE> SPN\n",
             f"{UNK_WORD} SPN\n"]

    with open(libri_lexicon, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                _line = line.strip().split("\t", maxsplit=1)
                word = _line[0]
                pronounceiation = _line[1]
            else:
                _line = line.strip().split(" ", maxsplit=1)
                word = _line[0]
                pronounceiation = _line[1]

            if word in keywords:
                assert word not in keyword_pron
                keyword_pron[word] = pronounceiation.strip()
                lines.append(line)

    with open(f"{in_dir}/lexicon.txt", "w", encoding="utf-8") as f:
        f.writelines(sorted(lines))

    if not os.path.exists("utils/prepare_lang.sh"):
        os.symlink(f"{KALDI_DIR}/egs/wsj/s5/utils", "utils")
        os.symlink(f"{KALDI_DIR}/egs/wsj/s5/steps", "steps")

    unk_fst_dir = "unk_fst"

    #####Optional  UNK FST
    # in librispeech_workdir
    # or reduce num ngram option
    ## using bigram only and num-ngrams is only 3336
    # num_extra_ngrams=1000
    run_shell(f"utils/lang/make_unk_lm.sh --num_extra_ngrams 1000 --ngram-order 2 {in_dir} {unk_fst_dir}")

    # TODO alternative simple phone loop

    prepare_lang_script = f"utils/prepare_lang.sh"
    run_shell(f"{prepare_lang_script} --unk-fst {unk_fst_dir}/unk_fst.txt {in_dir} \"<UNK>\" {tmpdir} {final_lang_dir}")

    # Grammar

    arpa_str = make_kw_arpa(keywords)
    with open("keyword.arpa", "w", encoding="utf-8") as f:
        f.writelines(arpa_str)

    run_shell(
        f"cat keyword.arpa | arpa2fst --disambig-symbol=#0 --read-symbol-table={final_lang_dir}/words.txt - {final_lang_dir}/G.fst")
    run_shell(f"utils/validate_lang.pl --skip-determinization-check {final_lang_dir}")

    final_graph_dir = "graph_final"

    run_shell(f"utils/mkgraph.sh {final_lang_dir} {train_graph_dir} {final_graph_dir}")

    return os.path.abspath(final_graph_dir)


if __name__ == '__main__':
    print(make_kaldi_decoding_graph())
