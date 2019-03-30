import os
import shutil

from kws_decoder.build_kw_grammar_fst import build_kw_grammar_fst
from kws_decoder.const_symbols import SIL_SYM, UNK_SYM, SPN_SYM
from utils.utils import run_shell

KALDI_ROOT = os.environ['KALDI_ROOT']


def check_andsetup__dirs(out_dir, train_graph_dir, train_dict_folder, lexicon_path):
    assert os.path.exists(f"{train_graph_dir}/final.mdl")
    assert os.path.exists(f"{train_graph_dir}/tree")
    # assert os.path.exists(f"{train_graph_dir}/frame_subsampling_factor") #TODO frame_subsampling_factor has to be defined here

    lang_in_tmp = os.path.join(out_dir, "lang_in_tmp")
    if not os.path.isdir(lang_in_tmp):
        os.makedirs(lang_in_tmp)
    lang_tmp = os.path.join(out_dir, "lang_tmp")
    if not os.path.isdir(lang_tmp):
        os.makedirs(lang_tmp)
    final_lang_dir = os.path.join(out_dir, "lang")
    if not os.path.isdir(final_lang_dir):
        os.makedirs(final_lang_dir)

    for static_file in ["extra_questions.txt", "nonsilence_phones.txt", "optional_silence.txt", "silence_phones.txt"]:
        if not os.path.exists(f"{lang_in_tmp}/{static_file}"):
            shutil.copy(f"{train_dict_folder}/{static_file}", f"{lang_in_tmp}/{static_file}")

    libri_lexicon = f"{lang_in_tmp}/librispeech-lexicon.txt"
    if not os.path.exists(libri_lexicon):
        shutil.copy(lexicon_path, libri_lexicon)

    return libri_lexicon, lang_in_tmp, lang_tmp, final_lang_dir


def filter_lexicon(keywords, libri_lexicon, out_folder):
    keywords = [kw.upper() for kw in keywords]

    lines = [f"{SIL_SYM} SIL\n",
             f"{SPN_SYM} SPN\n",
             f"{UNK_SYM} SPN\n"]

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

    with open(f"{out_folder}/lexicon.txt", "w", encoding="utf-8") as f:
        f.writelines(sorted(lines))


def make_kaldi_decoding_graph(keywords, out_dir,
                              train_graph_dir=f"{KALDI_ROOT}/egs/librispeech/s5/exp/tri4b",
                              train_dict_folder=f"{KALDI_ROOT}/egs/librispeech/s5/data/local/dict_nosp",
                              lexicon_path=f"{KALDI_ROOT}/egs/librispeech/s5/data/local/lm/librispeech-lexicon.txt",
                              draw_G_L_fsts=True):
    libri_lexicon, lang_in_tmp, lang_tmp, final_lang_dir = \
        check_andsetup__dirs(out_dir, train_graph_dir, train_dict_folder, lexicon_path)

    keywords = [kw.upper() for kw in keywords]

    if not os.path.exists(os.path.join(out_dir, "utils/prepare_lang.sh")):
        os.symlink(f"{KALDI_ROOT}/egs/wsj/s5/utils", os.path.join(out_dir, "utils"))
        os.symlink(f"{KALDI_ROOT}/egs/wsj/s5/steps", os.path.join(out_dir, "steps"))

    filter_lexicon(keywords, libri_lexicon, out_folder=lang_in_tmp)

    # TODO explore unk fst

    # unk_fst_dir = os.path.join(out_dir, "unk_fst")
    # if not os.path.isdir(unk_fst_dir):
    #     os.makedirs(unk_fst_dir)

    #####Optional  UNK FST
    # in librispeech_workdir
    # or reduce num ngram option
    ## using bigram only and num-ngrams is only 3336
    # num_extra_ngrams=1000

    # run_shell(
    #     f"{out_dir}/utils/lang/make_unk_lm.sh --num_extra_ngrams 1000 --ngram-order 2 --cmd utils/run.pl {lang_in_tmp} {unk_fst_dir}")

    # TODO alternative simple phone loop

    cwd = os.getcwd()
    os.chdir(out_dir)  # necessary because the kaldi scripts expect it

    if not os.path.exists("path.sh"):
        with open("path.sh", "w") as f:
            f.writelines("\n".join(["""export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH""",
                                    """[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1""",
                                    """. $KALDI_ROOT/tools/config/common_path.sh""",
                                    """export LC_ALL=C""", ""]))

    prepare_lang_script = f"{out_dir}/utils/prepare_lang.sh"
    # run_shell(f"{prepare_lang_script} --unk-fst {unk_fst_dir}/unk_fst.txt {lang_in_tmp} \"{unk_sym}\" {lang_tmp} {final_lang_dir}")
    run_shell(f"{prepare_lang_script} {lang_in_tmp} \"{UNK_SYM}\" {lang_tmp} {final_lang_dir}")

    if draw_G_L_fsts:
        run_shell(
            f"fstdraw --isymbols={final_lang_dir}/phones.txt "
            + f"--osymbols={final_lang_dir}/words.txt {final_lang_dir}/L.fst | dot -Tpdf -o{out_dir}/L.pdf")

    grammar_fst_path = build_kw_grammar_fst(keywords, words_file=f"{final_lang_dir}/words.txt")
    shutil.copy(grammar_fst_path, f"{final_lang_dir}/G.fst")

    if draw_G_L_fsts:
        run_shell(
            f"fstdraw --isymbols={final_lang_dir}/words.txt "
            + f"--osymbols={final_lang_dir}/words.txt {final_lang_dir}/G.fst | dot -Tpdf -o{out_dir}/G.pdf")

    run_shell(f"{out_dir}/utils/validate_lang.pl --skip-determinization-check {final_lang_dir}")

    final_graph_dir = os.path.join(out_dir, "graph_final")
    if not os.path.isdir(final_graph_dir):
        os.makedirs(final_graph_dir)

    run_shell(f"{out_dir}/utils/mkgraph.sh {final_lang_dir} {train_graph_dir} {final_graph_dir}")

    if not os.path.exists(os.path.join(final_graph_dir, "final.mdl")):
        os.symlink(f"{train_graph_dir}/final.mdl", os.path.join(final_graph_dir, "final.mdl"))

    os.chdir(cwd)

    return os.path.abspath(final_graph_dir)
