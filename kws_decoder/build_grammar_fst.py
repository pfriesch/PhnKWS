import os

from utils.utils import run_shell


def build_grammar_fst(arpa_lm_path, graph_dir):
    eesen_utils_path = os.path.join(os.getcwd(), "kws_decoder", "eesen_utils")

    assert os.path.exists(f"{eesen_utils_path}/s2eps.pl")
    assert os.access(f"{eesen_utils_path}/s2eps.pl", os.X_OK)

    assert os.path.exists(f"{eesen_utils_path}/eps2disambig.pl")
    assert os.access(f"{eesen_utils_path}/eps2disambig.pl", os.X_OK)

    run_shell(f"    gunzip -c {arpa_lm_path} | "
              + f"grep -v '<s> <s>' | "
              + f" grep -v '</s> <s>' |"
              + f"grep -v '</s> </s>' |"
              + f"arpa2fst - | fstprint | "
              # + f"utils/remove_oovs.pl {tmpdir}/oovs_{lm_suffix}.txt | "
              + f"{eesen_utils_path}/eps2disambig.pl | {eesen_utils_path}/s2eps.pl | fstcompile --isymbols={graph_dir}/words.txt "
              + f"--osymbols={graph_dir}/words.txt  --keep_isymbols=false --keep_osymbols=false | "
              + f"fstrmepsilon | fstarcsort --sort_type=ilabel > {graph_dir}/G.fst")

    return os.path.abspath(f"{graph_dir}/G.fst")
