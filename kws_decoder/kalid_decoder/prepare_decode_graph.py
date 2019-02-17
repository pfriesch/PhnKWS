import os
import shutil

from kws_decoder.arpa_utils import make_kw_arpa, UNK_WORD
from kws_decoder.kalid_decoder.build_fst_sh import build_kw_grammar_fst
from utils.logger_config import logger
from utils.utils import run_shell

KALDI_DIR = "/mnt/data/libs/kaldi"


def make_kaldi_decoding_graph(keywords, out_dir):
    # assert os.getcwd().endswith("kalid_decoder")  # TODO handle the relative folders

    train_graph_dir = f"{KALDI_DIR}/egs/librispeech/s5/exp/tri4b"
    train_dict_folder = f"{KALDI_DIR}/egs/librispeech/s5/data/local/dict_nosp"
    lexicon_path = f"{KALDI_DIR}/egs/librispeech/s5/data/local/lm/librispeech-lexicon.txt"

    in_dir = os.path.join(out_dir, "in_tmp")
    if not os.path.isdir(in_dir):
        os.makedirs(in_dir)
    tmpdir = os.path.join(out_dir, "tmp")
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)
    final_lang_dir = os.path.join(out_dir, "lang")
    if not os.path.isdir(final_lang_dir):
        os.makedirs(final_lang_dir)

    if not hasattr(logger, "logger"):
        logger.configure_logger(tmpdir)

    for static_file in ["extra_questions.txt", "nonsilence_phones.txt", "optional_silence.txt", "silence_phones.txt"]:
        if not os.path.exists(f"{in_dir}/{static_file}"):
            shutil.copy(f"{train_dict_folder}/{static_file}", f"{in_dir}/{static_file}")

    libri_lexicon = f"{in_dir}/librispeech-lexicon.txt"
    if not os.path.exists(libri_lexicon):
        shutil.copy(lexicon_path, libri_lexicon)

    keywords = [kw.upper() for kw in keywords]

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
                lines.append(line)

    with open(f"{in_dir}/lexicon.txt", "w", encoding="utf-8") as f:
        f.writelines(sorted(lines))

    if not os.path.exists(os.path.join(out_dir, "utils/prepare_lang.sh")):
        os.symlink(f"{KALDI_DIR}/egs/wsj/s5/utils", os.path.join(out_dir, "utils"))
        os.symlink(f"{KALDI_DIR}/egs/wsj/s5/steps", os.path.join(out_dir, "steps"))

    # unk_fst_dir = os.path.join(out_dir, "unk_fst")
    # if not os.path.isdir(unk_fst_dir):
    #     os.makedirs(unk_fst_dir)

    #####Optional  UNK FST
    # in librispeech_workdir
    # or reduce num ngram option
    ## using bigram only and num-ngrams is only 3336
    # num_extra_ngrams=1000

    cwd = os.getcwd()
    os.chdir(out_dir)

    if not os.path.exists("path.sh"):
        with open("path.sh", "w") as f:
            f.writelines("\n".join(["""export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH""",
                                    """[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1""",
                                    """. $KALDI_ROOT/tools/config/common_path.sh""",
                                    """export LC_ALL=C""", ""]))

    # run_shell(
    #     f"{out_dir}/utils/lang/make_unk_lm.sh --num_extra_ngrams 1000 --ngram-order 2 --cmd utils/run.pl {in_dir} {unk_fst_dir}")

    # TODO alternative simple phone loop

    prepare_lang_script = f"{out_dir}/utils/prepare_lang.sh"
    # run_shell(f"{prepare_lang_script} --unk-fst {unk_fst_dir}/unk_fst.txt {in_dir} \"<UNK>\" {tmpdir} {final_lang_dir}")
    run_shell(f"{prepare_lang_script} {in_dir} \"<UNK>\" {tmpdir} {final_lang_dir}")

    # if viz_model:#TODO add flag
    run_shell(
        f"fstdraw --isymbols={final_lang_dir}/phones.txt --osymbols={final_lang_dir}/words.txt {final_lang_dir}/L.fst | dot -Tpdf -o{out_dir}/L.pdf")

    # Grammar

    # arpa_str = make_kw_arpa(keywords)
    # with open(f"{out_dir}/keyword.arpa", "w", encoding="utf-8") as f:
    #     f.writelines(arpa_str)

    grammar_fst_path = build_kw_grammar_fst(keywords, words_file=f"{final_lang_dir}/words.txt")
    shutil.copy(grammar_fst_path, f"{final_lang_dir}/G.fst")
    # try:
    #     run_shell(
    #         f"cat {out_dir}/keyword.arpa | arpa2fst --disambig-symbol=#0 --read-symbol-table={final_lang_dir}/words.txt - {final_lang_dir}/G.fst")
    # except RuntimeError:
    #     with open(f"{out_dir}/keyword.arpa", "r") as f:
    #         print("".join(f.readlines()))
    #     raise

    # if viz_model:#TODO add flag
    run_shell(
        f"fstdraw --isymbols={final_lang_dir}/words.txt --osymbols={final_lang_dir}/words.txt {final_lang_dir}/G.fst | dot -Tpdf -o{out_dir}/G.pdf")

    run_shell(f"{out_dir}/utils/validate_lang.pl --skip-determinization-check {final_lang_dir}")

    final_graph_dir = os.path.join(out_dir, "graph_final")
    if not os.path.isdir(final_graph_dir):
        os.makedirs(final_graph_dir)

    run_shell(f"{out_dir}/utils/mkgraph.sh {final_lang_dir} {train_graph_dir} {final_graph_dir}")

    if not os.path.exists(os.path.join(final_graph_dir, "final.mdl")):
        os.symlink(f"{train_graph_dir}/final.mdl", os.path.join(final_graph_dir, "final.mdl"))

    os.chdir(cwd)

    return os.path.abspath(final_graph_dir)
