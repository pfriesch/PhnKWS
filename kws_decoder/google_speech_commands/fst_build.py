#!/usr/bin/env python
import os
import sys

from utils.utils import run_shell


def ctc_token_fst(lines):
    out_lines = ['0 1 <eps> <eps>',
                 '1 1 <blk> <eps>',
                 '2 2 <blk> <eps>',
                 '2 0 <eps> <eps>']

    nodeX = 3
    for entry in lines:
        entry = entry.replace('\n', '').strip()
        fields = entry.split(' ')
        phone = fields[0]
        if phone == '<eps>' or phone == '<blk>':
            continue

        if '#' in phone:
            out_lines.append(f"0 0 <eps> {phone}")
        else:
            out_lines.append(f"1 {nodeX} {phone} {phone}")
            out_lines.append(f"{nodeX} {nodeX} {phone} <eps>")
            out_lines.append(f"{nodeX} 2 <eps> <eps>")
        nodeX += 1
    out_lines.append('0')

    return "\n".join(out_lines)


def check_environment():
    # Run shich to check if binaries are in PATH
    run_shell("which fstcompile")
    run_shell("which fstaddselfloops")
    run_shell("which fstarcsort")

    run_shell("which fsttablecompose")
    run_shell("which fstdeterminizestar")
    run_shell("which fstminimizeencoded")

    run_shell("which arpa2fst")


def build_fst(lm_data_folder):
    keyword_tokens_file = os.path.join(lm_data_folder, "keyword_tokens.txt")
    assert os.path.exists(keyword_tokens_file)
    ctc_keyword_tokens_file = os.path.join(lm_data_folder, "keyword_tokens_ctc.txt")
    assert os.path.exists(ctc_keyword_tokens_file)
    keywords_path = os.path.join(lm_data_folder, "keywords.txt")
    assert os.path.exists(keywords_path)
    keyword_lexicon_p_path = os.path.join(lm_data_folder, "keyword_lexiconp.txt")
    assert os.path.exists(keyword_lexicon_p_path)
    keyword_lexicon_p_disambig_path = os.path.join(lm_data_folder, "keyword_lexiconp_disambig.txt")
    assert os.path.exists(keyword_lexicon_p_disambig_path)
    keyword_arpa = os.path.join(lm_data_folder, "keyword.arpa")
    assert os.path.exists(keyword_arpa)

    fst_out_path = os.path.join(sys.modules['__main__'].__file__, "../fst/")  # TODO find folder in exp dir

    script_folder = os.path.join(sys.modules['__main__'].__file__, "../scripts/")

    T_fst_path = f"{fst_out_path}/T.fst"

    with open(keyword_tokens_file, "r") as f:
        tokens = f.readlines()

    ctc_tokens = ctc_token_fst(tokens)
    with open(ctc_keyword_tokens_file, "w") as f:
        f.writelines(ctc_tokens)

    run_shell(
        f"cat {ctc_keyword_tokens_file} | "
        + f"fstcompile "
        + f"--isymbols={keyword_tokens_file} "
        + f"--osymbols={keyword_tokens_file} "
        + f"--keep_isymbols=false --keep_osymbols=false | "
        + f"fstarcsort --sort_type=olabel > {T_fst_path} || exit 1;")

    L_fst_path = f"{fst_out_path}/L.fst"

    token_disambig_symbol = int(run_shell(f"`grep \#0 {keyword_tokens_file} | awk '{{print $2}}'`").decode("utf-8"))
    word_disambig_symbol = int(run_shell(f"`grep \#0 {keywords_path} | awk '{{print $2}}'`").decode("utf-8"))
    assert token_disambig_symbol == 40
    assert word_disambig_symbol == 42

    add_lex_disambig_tool_path = os.path.join(script_folder, "utils/add_lex_disambig.pl")

    ndisambig = run_shell(f"{add_lex_disambig_tool_path} {keyword_lexicon_p_path} {keyword_lexicon_p_disambig_path}")
    ndisambig = int(ndisambig.decode("utf-8").strip())
    ndisambig += 1

    make_lexicon_fst_tool_path = os.path.join(script_folder, "utils/make_lexicon_fst.pl")

    # lexicon_fst = "asr_egs/librispeech/config/keyword_lexicon_fst.txt"
    # run_shell(
    #     f"{make_lexicon_fst_tool_path} --pron-probs {keyword_lexicon_p_disambig_path} 0 \"sil\" '#'{ndisambig} > {lexicon_fst}")

    run_shell(f"{make_lexicon_fst_tool_path} --pron-probs {keyword_lexicon_p_disambig_path} 0 \"sil\" '#'{ndisambig} | "
              + f"fstcompile --isymbols={keyword_tokens_file} --osymbols={keywords_path} "
              + f"--keep_isymbols=false --keep_osymbols=false | "
              + f"fstaddselfloops  \"echo {token_disambig_symbol} |\" \"echo {word_disambig_symbol} |\" | "
              + f"fstarcsort --sort_type=olabel > {L_fst_path} ")

    # keyword_arpa = "asr_egs/librispeech/config/keyword.arpa"
    eps_disambig_tool_path = os.path.join(script_folder, "utils/eps2disambig.pl")
    s2eps_tool_path = os.path.join(script_folder, "utils/s2eps.pl")

    G_fst_path = f"{fst_out_path}/G.fst"

    run_shell(f"cat {keyword_arpa} | arpa2fst - | fstprint | " \
              + f"{eps_disambig_tool_path} | {s2eps_tool_path} | fstcompile --isymbols={keywords_path} " \
              + f"--osymbols={keywords_path} --keep_isymbols=false --keep_osymbols=false | " \
              + f"fstrmepsilon | fstarcsort --sort_type=ilabel > {G_fst_path}")

    LG_fst_path = f"{fst_out_path}/LG.fst"
    TLG_fst_path = f"{fst_out_path}/TLG.fst"

    # Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
    # minimized.
    # os.environ["PATH"]

    run_shell(f"fsttablecompose {L_fst_path} {G_fst_path} | fstdeterminizestar --use-log=true | " \
              + f"fstminimizeencoded | fstarcsort --sort_type=ilabel > {LG_fst_path} || exit 1;")
    run_shell(f"fsttablecompose {T_fst_path} {LG_fst_path} > {TLG_fst_path} || exit 1;")
