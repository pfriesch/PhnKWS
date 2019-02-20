import os
import shutil

import torch

from data.phoneme_dict import get_phoneme_dict
from kws_decoder.build_fst import build_kw_grammar_fst
from kws_decoder.const_symbols import SIL_SYM, UNK_SYM, SPN_SYM
from utils.utils import run_shell

KALDI_ROOT = os.environ['KALDI_ROOT']  # "/mnt/data/libs/kaldi"


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


def rm_stress_marks(pronounceiation):
    for i in range(3):
        pronounceiation = pronounceiation.replace(str(i), "")
    return pronounceiation


def filter_lexicon(keywords, libri_lexicon, out_lexicon):
    lines = [f"{SIL_SYM} SIL\n",
             f"{SPN_SYM} SPN\n",
             f"{UNK_SYM} SPN\n"]

    with open(libri_lexicon, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                _line = line.split("\t", maxsplit=1)
                word = _line[0]
                pronounceiation = _line[1]
            else:
                _line = line.split(" ", maxsplit=1)
                word = _line[0]
                pronounceiation = _line[1]

            if word in keywords:
                lines.append(word + " " + rm_stress_marks(pronounceiation))

    with open(out_lexicon, "w", encoding="utf-8") as f:
        f.writelines(sorted(lines))


def main(keywords, lexicon_path, phn2idx, draw_G_L_fsts=False):
    tmpdir = "tmp"
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir)

    graph_dir = "graph_dir"
    if os.path.isdir(graph_dir):
        shutil.rmtree(graph_dir)
    os.makedirs(graph_dir)

    keywords = [kw.upper() for kw in keywords]

    #######

    # check_units_txt(units_txt)
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    assert os.path.exists(lexicon_path)

    filter_lexicon(keywords, lexicon_path, f"{tmpdir}/lexicon.txt")
    lexicon_path = f"{tmpdir}/lexicon.txt"

    # Add probabilities to lexicon entries. There is in fact no point of doing this here since all the entries have 1.0.
    # But utils/make_lexicon_fst.pl requires a probabilistic version, so we just leave it as it is. 
    # run_shell(f"perl -ape 's/(\S+\s+)(.+)/${{1}}1.0\\t$2/;' < {lexicon_path} > {tmpdir}/lexiconp.txt")

    # Add disambiguation symbols to the lexicon. This is necessary for determinizing the composition of L.fst and G.fst.
    # Without these symbols, determinization will fail. 

    assert os.path.exists("utils/add_lex_disambig.pl")
    assert os.access("utils/add_lex_disambig.pl", os.X_OK)
    ndisambig = int(
        run_shell(f"utils/add_lex_disambig.pl {tmpdir}/lexicon.txt {tmpdir}/lexicon_disambig.txt").strip())
    assert isinstance(ndisambig, int)
    ndisambig += 1

    with open(f"{tmpdir}/disambig.list", "w") as f:
        f.writelines([f"#{n}\n" for n in range(ndisambig)])

    # Get the full list of CTC tokens used in FST. These tokens include <eps>, the blank <blk>, the actual labels (e.g.,
    # phonemes), and the disambiguation symbols.

    with open(f"{tmpdir}/units.list", "w") as f:
        f.write("SPN\n")
        f.write("UNK\n")
        for phn in phn2idx:
            # if "SIL" not in phn and "sil" not in phn:
            f.write(f"{phn.upper()}\n")
        # f.write(f"\n")

    # run_shell(f"cat {units_txt} | awk '{{print $1}}' > {tmpdir}/units.list")
    run_shell(f"(echo '<eps>'; echo '<blk>';) | cat - {tmpdir}/units.list {tmpdir}/disambig.list "
              + f"| awk '{{print $1 \" \" (NR-1)}}' > {graph_dir}/tokens.txt")

    with open(f"{graph_dir}/tokens.txt", "r") as f:
        token_lines = f.readlines()
    toekn_fst_txt = ctc_token_fst(token_lines)
    with open(f"{graph_dir}/tokens_fst.txt", "w") as f:
        f.writelines(toekn_fst_txt)

    run_shell(f"cat {graph_dir}/tokens_fst.txt | fstcompile --isymbols={graph_dir}/tokens.txt --osymbols={graph_dir}/tokens.txt \
       --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > {graph_dir}/T.fst ")

    # Encode the words with indices. Will be used in lexicon and language model FST compiling.
    run_shell(f"""
    cat {tmpdir}/lexicon.txt | awk '{{print $1}}' | sort | uniq  | awk '
      BEGIN {{
        print "<eps> 0";
      }} 
      {{
        printf("%s %d\\n", $1, NR);
      }}
      END {{
        printf("#0 %d\\n", NR+1);
      }}' > {graph_dir}/words.txt || exit 1;
    """)

    # Now compile the lexicon FST. Depending on the size of your lexicon, it may take some time.
    token_disambig_symbol = int(run_shell(f"grep \#0 {graph_dir}/tokens.txt | awk '{{print $2}}'").strip())
    word_disambig_symbol = int(run_shell(f"grep \#0 {graph_dir}/words.txt | awk '{{print $2}}'").strip())

    # TODO why does piping not work?
    lexicon_fst = run_shell(f"utils/make_lexicon_fst.pl {tmpdir}/lexicon_disambig.txt 0 \"SIL\" #{ndisambig}")

    run_shell(f"echo \"{lexicon_fst}\" | "
              + f"fstcompile --isymbols={graph_dir}/tokens.txt --osymbols={graph_dir}/words.txt "
              + f"--keep_isymbols=false --keep_osymbols=false | "
              + f"fstaddselfloops  \"echo {token_disambig_symbol} |\" \"echo {word_disambig_symbol} |\" | "
              + f"fstarcsort --sort_type=olabel > {graph_dir}/L.fst")

    if draw_G_L_fsts:
        run_shell(
            f"fstdraw --isymbols={graph_dir}/tokens.txt "
            + f"--osymbols={graph_dir}/words.txt {graph_dir}/L.fst | dot -Tpdf -o{graph_dir}/L.pdf")

    ########## MkGraph

    grammar_fst_path = build_kw_grammar_fst(keywords, words_file=f"{graph_dir}/words.txt")
    shutil.copy(grammar_fst_path, f"{graph_dir}/G.fst")

    if draw_G_L_fsts:
        run_shell(
            f"fstdraw --isymbols={graph_dir}/words.txt "
            + f"--osymbols={graph_dir}/words.txt {graph_dir}/G.fst | dot -Tpdf -o{graph_dir}/G.pdf")

    run_shell(f"fsttablecompose {graph_dir}/L.fst {graph_dir}/G.fst | fstdeterminizestar --use-log=true | "
              + f"fstminimizeencoded | fstarcsort --sort_type=ilabel > {graph_dir}/LG.fst")
    run_shell(f"fsttablecompose {graph_dir}/T.fst {graph_dir}/LG.fst > {graph_dir}/TLG.fst")

    return os.path.abspath(graph_dir)


if __name__ == '__main__':
    config = \
        torch.load("/mnt/data/pytorch-kaldi/exp/TIMIT_MLP_fbank_20190219_172928/checkpoints/checkpoint-epoch33.pth",
                   map_location='cpu')['config']
    phoneme_dict = get_phoneme_dict(config['dataset']['dataset_definition']['phn_mapping_file'],
                                    stress_marks=True, word_position_dependency=False)

    print((main(keywords=['alexa', 'left', 'right'],
                lexicon_path=f"{KALDI_ROOT}/egs/librispeech/s5/data/local/lm/librispeech-lexicon.txt",
                phn2idx=phoneme_dict.phoneme2reducedIdx, draw_G_L_fsts=True)))
