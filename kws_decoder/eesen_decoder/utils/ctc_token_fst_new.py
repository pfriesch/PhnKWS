#!/usr/bin/env python
import os

from asr_egs.librispeech.sh_utils import run_shell


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


if __name__ == '__main__':
    run_shell("which fstcompile")
    run_shell("which fstaddselfloops")
    run_shell("which fstarcsort")

    run_shell("which fsttablecompose")
    run_shell("which fstdeterminizestar")
    run_shell("which fstminimizeencoded")

    run_shell("which arpa2fst")

    fstaddselfloops_path = "$KALDI_ROOT/src/fstbin/fstaddselfloops"

    os.chdir('/mnt/data/eesen')
    tokens_path = "asr_egs/librispeech/config/keyword_tokens.txt"
    ctc_tokens_path = "asr_egs/librispeech/config/keyword_tokens_ctc.txt"
    T_fst_path = "asr_egs/librispeech/exp/nml_seq_fw_seq_tw/lang/T.fst"

    with open(tokens_path, "r") as f:
        tokens = f.readlines()

    ctc_tokens = ctc_token_fst(tokens)
    with open(ctc_tokens_path, "w") as f:
        f.writelines(ctc_tokens)

    run_shell(
        f"cat {ctc_tokens_path} | "
        + f"fstcompile "
        + f"--isymbols={tokens_path} "
        + f"--osymbols={tokens_path} "
        + f"--keep_isymbols=false --keep_osymbols=false | "
        + f"fstarcsort --sort_type=olabel > {T_fst_path} || exit 1;")

    keywords_path = "asr_egs/librispeech/config/keywords.txt"
    keyword_lexicon_p_path = "asr_egs/librispeech/config/keyword_lexiconp.txt"
    keyword_lexicon_p_disambig_path = "asr_egs/librispeech/config/keyword_lexiconp_disambig.txt"
    L_fst_path = "asr_egs/librispeech/exp/nml_seq_fw_seq_tw/lang/L.fst"

    token_disambig_symbol = 40  # TODO index of #0 in tokens_path
    word_disambig_symbol = 42  # TODO index of #0 in keywords_path

    # token_disambig_symbol=`grep \#0 asr_egs/librispeech/config/keyword_tokens.txt | awk '{print $2}'`

    add_lex_disambig_tool_path = "asr_egs/librispeech/utils/add_lex_disambig.pl"

    ndisambig = run_shell(f"{add_lex_disambig_tool_path} {keyword_lexicon_p_path} {keyword_lexicon_p_disambig_path}")
    ndisambig = int(ndisambig.decode("utf-8").strip())
    ndisambig += 1

    make_lexicon_fst_tool_path = "asr_egs/librispeech/utils/make_lexicon_fst.pl"
    # lexicon_fst = "asr_egs/librispeech/config/keyword_lexicon_fst.txt"
    # run_shell(
    #     f"{make_lexicon_fst_tool_path} --pron-probs {keyword_lexicon_p_disambig_path} 0 \"sil\" '#'{ndisambig} > {lexicon_fst}")

    run_shell(f"{make_lexicon_fst_tool_path} --pron-probs {keyword_lexicon_p_disambig_path} 0 \"sil\" '#'{ndisambig} | "
              + f"fstcompile --isymbols={tokens_path} --osymbols={keywords_path} "
              + f"--keep_isymbols=false --keep_osymbols=false | "
              + f"{fstaddselfloops_path}  \"echo {token_disambig_symbol} |\" \"echo {word_disambig_symbol} |\" | "
              + f"fstarcsort --sort_type=olabel > {L_fst_path} ")

    keyword_arpa = "asr_egs/librispeech/config/keyword.arpa"
    eps_disambig_tool_path = "asr_egs/librispeech/utils/eps2disambig.pl"
    s2eps_tool_path = "asr_egs/librispeech/utils/s2eps.pl"
    G_fst_path = "asr_egs/librispeech/exp/nml_seq_fw_seq_tw/lang/G.fst"

    run_shell(f"cat {keyword_arpa} | arpa2fst - | fstprint | " \
              + f"{eps_disambig_tool_path} | {s2eps_tool_path} | fstcompile --isymbols={keywords_path} " \
              + f"--osymbols={keywords_path} --keep_isymbols=false --keep_osymbols=false | " \
              + f"fstrmepsilon | fstarcsort --sort_type=ilabel > {G_fst_path}")

    LG_fst_path = "asr_egs/librispeech/exp/nml_seq_fw_seq_tw/lang/LG.fst"
    TLG_fst_path = "asr_egs/librispeech/exp/nml_seq_fw_seq_tw/lang/TLG.fst"

    # Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
    # minimized.
    # os.environ["PATH"]

    run_shell(f"fsttablecompose {L_fst_path} {G_fst_path} | fstdeterminizestar --use-log=true | " \
              + f"fstminimizeencoded | fstarcsort --sort_type=ilabel > {LG_fst_path} || exit 1;")
    run_shell(f"fsttablecompose {T_fst_path} {LG_fst_path} > {TLG_fst_path} || exit 1;")
