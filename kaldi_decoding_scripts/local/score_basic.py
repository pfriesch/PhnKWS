import os

from utils.utils import run_shell


def score(data, lang_or_graph, dir, num_jobs,
          min_lmwt=1,
          max_lmwt=10):
    decoing_scripts_folder = os.path.join(os.getcwd(), __name__.split(".")[0])  # 'kaldi_decoding_scripts'

    # end configuration section.

    phonemap = "conf/phones.60-48-39.map"
    nj = num_jobs

    symtab = os.path.join(lang_or_graph, "words.txt")

    assert os.path.exists(symtab)
    assert os.path.exists(os.path.join(dir, "lat.1.gz"))
    assert os.path.exists(os.path.join(data, "text"))
    timit_norm_trans_script = os.path.join(decoing_scripts_folder, "local/timit_norm_trans.pl")
    assert os.path.exists(timit_norm_trans_script)
    assert os.access(timit_norm_trans_script, os.X_OK)
    int2sym_script = os.path.join(decoing_scripts_folder, "utils/int2sym.pl")
    assert os.path.exists(int2sym_script)
    assert os.access(int2sym_script, os.X_OK)
    phonemap = os.path.join(decoing_scripts_folder, phonemap)
    assert os.path.exists(phonemap)
    pl_cmd_script = os.path.join(decoing_scripts_folder, "utils/run.pl")
    assert os.path.exists(pl_cmd_script)
    assert os.access(pl_cmd_script, os.X_OK)

    os.makedirs(os.path.join(dir, "scoring", "log"))

    # Map reference to 39 phone classes:
    cmd = f"cat {data}/text | {timit_norm_trans_script} -i - -m {phonemap} -from 48 -to 39 > {dir}/scoring/test_filt.txt"
    run_shell(cmd)

    # Get the phone-sequence on the best-path:
    for LMWT in range(min_lmwt, max_lmwt):
        cmd = f"{pl_cmd_script} JOB=1:{nj} {dir}/scoring/log/best_path_basic.{LMWT}.JOB.log " + \
              f"lattice-best-path --lm-scale={LMWT} --word-symbol-table={symtab} --verbose=2 \"ark:gunzip -c {dir}/lat.JOB.gz|\" ark,t:{dir}/scoring/{LMWT}.JOB.tra || exit 1;"
        run_shell(cmd)
        run_shell(f"cat {dir}/scoring/{LMWT}.*.tra | sort > {dir}/scoring/{LMWT}.tra")
        run_shell(f"rm {dir}/scoring/{LMWT}.*.tra")

    # Map hypothesis to 39 phone classes:
    cmd = f"{pl_cmd_script} LMWT={min_lmwt}:{max_lmwt}{dir}/scoring/log/score_basic.LMWT.log " + \
          f"cat {dir}/scoring/LMWT.tra \| " + \
          f"{int2sym_script} -f 2- {symtab} \| " + \
          f"{timit_norm_trans_script} -i - -m {phonemap} -from 48 -to 39 \| " + \
          f"compute-wer --text --mode=all ark:{dir}/scoring/test_filt.txt ark,p:- \">&\" {dir}/wer_LMWT || exit 1;"
    run_shell(cmd)
