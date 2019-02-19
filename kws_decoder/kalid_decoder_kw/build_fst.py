import os
import shutil
from glob import glob

from kws_decoder.const_symbols import EPS_SYM, SIL_SYM, UNK_SYM, SPN_SYM
from utils.utils import run_shell


def write_fst(kw, keyword_fst_folder, word_map):
    w = word_map

    fst_str = []
    idx = 0
    for word in kw:
        fst_str.append(f"{idx} {idx + 1} {w[word]} {w[word]}")
        idx += 1
    fst_str.append(f"{idx}")
    fst_str.append(f"")
    with open(f"{keyword_fst_folder}/{'_'.join([w.replace('<', '').replace('>', '') for w in kw])}.txt", "w") as f:
        f.writelines("\n".join(fst_str))


def build_kw_grammar_fst(keywords, words_file):
    if not isinstance(keywords[0], list):
        # expect each kw to be a list of words
        keywords = [kw.split(" ") for kw in keywords]

    with open(words_file, "r") as f:
        word_map = f.readlines()
        word_map = dict([line.strip().split(" ", 1) for line in word_map])
        assert EPS_SYM in word_map
        assert SIL_SYM in word_map
        assert UNK_SYM in word_map
        assert SPN_SYM in word_map

    keyword_fst_folder = "keyword_fsts"
    if os.path.isdir(keyword_fst_folder):
        shutil.rmtree(keyword_fst_folder)
    os.makedirs(keyword_fst_folder)

    write_fst([UNK_SYM], keyword_fst_folder, word_map)

    for kw in keywords:
        write_fst(kw, keyword_fst_folder, word_map)

    out_fst = f"{keyword_fst_folder}/UNION.fst"
    run_shell(f"fstcompile {keyword_fst_folder}/UNK.txt {out_fst}")

    for fst in glob(f"{keyword_fst_folder}/*.txt"):
        run_shell(f"fstcompile {fst} {fst[:-4]}.fst")
        run_shell(f"fstunion {fst[:-4]}.fst {out_fst} {out_fst}")

    run_shell(f"fstrmepsilon {out_fst} | fstarcsort --sort_type=ilabel - {out_fst}")

    # run_shell(f"fstprint --isymbols={words_file} --osymbols={words_file} {out_fst}")
    # run_shell(f"fstprint {out_fst}")
    # run_shell(
    #     f"fstdraw --isymbols={words_file} --osymbols={words_file} {out_fst} "
    #     + f"| dot -Tpdf -o {os.path.basename(out_fst)[:-4]}.pdf")

    return os.path.abspath(out_fst)
