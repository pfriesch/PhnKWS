import os
import shutil
from glob import glob

from kws_decoder.const_symbols import EPS_SYM, SIL_SYM, UNK_SYM, SPN_SYM
from utils.utils import run_shell
import gzip


def build_1_gram_fst(arpa_lm_path, graph_dir):
    one_gram_arpa = []

    with gzip.open(arpa_lm_path, 'rb') as f:
        for line in f:
            line = line.decode("utf-8")
            if "\\2-grams:" in line:
                break
            elif not ("ngram 2=" in line or "ngram 3=" in line or "ngram 4=" in line
                      or "ngram 5=" in line or "ngram 6=" in line):
                if line.startswith("-"):
                    # removing backtracking prob for 1-gram
                    fake_prob = "-1.0"
                    word = line.split(maxsplit=2)[1]

                    _line = f"{fake_prob}\t{word}\n"
                else:
                    _line = line
                one_gram_arpa.append(_line)

            else:
                pass

    one_gram_arpa.append("\\end\\\n")
    pruned_lexicon_path = f"{graph_dir}/pruned_lexicon.txt"
    with open(pruned_lexicon_path, 'w') as f:
        f.writelines(one_gram_arpa)

    eesen_utils_path = os.path.join(os.getcwd(), "kws_decoder", "eesen_utils")

    assert os.path.exists(f"{eesen_utils_path}/s2eps.pl")
    assert os.access(f"{eesen_utils_path}/s2eps.pl", os.X_OK)

    assert os.path.exists(f"{eesen_utils_path}/eps2disambig.pl")
    assert os.access(f"{eesen_utils_path}/eps2disambig.pl", os.X_OK)

    run_shell(f"arpa2fst {pruned_lexicon_path} | fstprint | "
              # + f"utils/remove_oovs.pl {tmpdir}/oovs_{lm_suffix}.txt | "
              + f"{eesen_utils_path}/eps2disambig.pl | {eesen_utils_path}/s2eps.pl | fstcompile --isymbols={graph_dir}/words.txt "
              + f"--osymbols={graph_dir}/words.txt  --keep_isymbols=false --keep_osymbols=false | "
              + f"fstrmepsilon | fstarcsort --sort_type=ilabel > {graph_dir}/G.fst")

    return os.path.abspath(f"{graph_dir}/G.fst")

    # run_shell(f"gunzip -c {arpa_lm_path} | "
    #           + f"grep -v '<s> <s>' | "
    #           + f"grep -v '</s> <s>' | "
    #           + f"grep -v '</s> </s>' | "
    #           + f"arpa2fst - | fstprint | "
    #           # + f"utils/remove_oovs.pl $tmpdir/oovs_${lm_suffix}.txt | "
    #           + f"{eesen_utils_path}/eps2disambig.pl | {eesen_utils_path}/s2eps.pl | fstcompile --isymbols={graph_dir}/words.txt "
    #           + f"--osymbols={graph_dir}/words.txt  --keep_isymbols=false --keep_osymbols=false | "
    #           + f"fstrmepsilon | fstarcsort --sort_type=ilabel > {graph_dir}/G.fst")
