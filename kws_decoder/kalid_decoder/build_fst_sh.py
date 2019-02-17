import itertools
import os
import shutil
from glob import glob

from utils.utils import run_shell, check_environment


# #KWS_001
# 0	1	72070	72070
# 1	2	117093	117093
# 2
#
# KWS_002
# 0	1	156223	156223
# 1	2	198047	198047
# 2
#
# KWS_003
# 0	1	33224	33224
# 1
#
# KWS_004
# 0	1	185382	185382
# 1
#
# KWS_005
# 0	1	194651	194651
# 1	2	62931	62931
# 2
#
# KWS_006
# 0	1	49768	49768
# 1	2	40836	40836
# 2

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

    eps_sym = "<eps>"
    sil_sym = "!SIL"
    unk_sym = "<UNK>"
    spn_sym = "<SPOKEN_NOISE>"

    with open(words_file, "r") as f:
        word_map = f.readlines()
        word_map = dict([line.strip().split(" ", 1) for line in word_map])
        assert eps_sym in word_map
        assert sil_sym in word_map
        assert unk_sym in word_map
        assert spn_sym in word_map

    keyword_fst_folder = "keyword_fsts"
    if os.path.isdir(keyword_fst_folder):
        shutil.rmtree(keyword_fst_folder)
    os.makedirs(keyword_fst_folder)

    write_fst([unk_sym], keyword_fst_folder, word_map)

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
