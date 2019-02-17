import os

from utils.utils import run_shell


def score(data, lang_or_graph, dir,
          word_ins_penalty=None,
          min_lmwt=7,
          max_lmwt=17,
          ):
    if word_ins_penalty is None:
        word_ins_penalty = [0.0, 0.5, 1.0]
    decoing_scripts_folder = os.path.join(os.getcwd(), __name__.split(".")[0])  # 'kaldi_decoding_scripts'
    pl_cmd_script = os.path.join(decoing_scripts_folder, "utils/run.pl")
    assert os.path.exists(pl_cmd_script)
    assert os.access(pl_cmd_script, os.X_OK)
    symtab = os.path.join(lang_or_graph, "words.txt")
    assert os.path.exists(symtab)
    assert os.path.exists(os.path.join(dir, "lat.1.gz"))
    assert os.path.exists(os.path.join(data, "text"))
    int2sym_script = os.path.join(decoing_scripts_folder, "utils/int2sym.pl")
    assert os.path.exists(int2sym_script)
    assert os.access(int2sym_script, os.X_OK)
    if not os.path.isdir(os.path.join(dir, "scoring", "log")):
        os.makedirs(os.path.join(dir, "scoring", "log"))

    run_shell(f"cat {data}/text | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' > {dir}/scoring/test_filt.txt")

    for wip in word_ins_penalty:
        cmd = f"{pl_cmd_script} LMWT={min_lmwt}:{max_lmwt} {dir}/scoring/log/best_path.LMWT.{wip}.log " + \
              f"lattice-scale --inv-acoustic-scale=LMWT \"ark:gunzip -c {dir}/lat.*.gz|\" ark:- \| " + \
              f"lattice-add-penalty --word-ins-penalty={wip} ark:- ark:- \| " + \
              f"lattice-best-path --word-symbol-table={symtab} ark:- ark,t:{dir}/scoring/LMWT.{wip}.tra || exit 1;"
        run_shell(cmd)

    for wip in word_ins_penalty:
        cmd = f"{pl_cmd_script} LMWT={min_lmwt}:{max_lmwt} {dir}/scoring/log/score.LMWT.{wip}.log " + \
              f"cat {dir}/scoring/LMWT.{wip}.tra \| " + \
              f"{int2sym_script} -f 2- {symtab} \| sed 's:\<UNK\>::g' \| " + \
              f"compute-wer --text --mode=present ark:{dir}/scoring/test_filt.txt  ark,p:- \">&\" {dir}/wer_LMWT_{wip} || exit 1;"
        run_shell(cmd)
