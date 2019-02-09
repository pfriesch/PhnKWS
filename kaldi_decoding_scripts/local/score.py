import os
from shutil import copy

from utils.utils import run_shell


def score(data, lang_or_graph, dir, num_jobs, stage=0,
          model=None,
          min_lmwt=1,
          max_lmwt=10,
          mbr_scale=1.0
          ):
    decoing_scripts_folder = os.path.join(os.getcwd(), __name__.split(".")[0])  # 'kaldi_decoding_scripts'
    # end configuration section.

    if model is None:
        # assume model one level up from decoding dir.
        model = os.path.join(dir, "..", "final.mdl")

    KALDI_ROOT = os.environ['KALDI_ROOT']

    hubscr = f"{KALDI_ROOT}/tools/sctk/bin/hubscr.pl"
    assert os.path.exists(f"{KALDI_ROOT}/tools/sctk/bin/hubscr.pl"), "Cannot find scoring program hubscr"

    hubdir = os.path.dirname(f"{KALDI_ROOT}/tools/sctk/bin/hubscr.pl")

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
    assert os.path.exists(pl_cmd_script) #TODO remove run.pl command
    assert os.access(pl_cmd_script, os.X_OK)

    os.makedirs(os.path.join(dir, "scoring", "log"))

    # Map reference to 39 phone classes, the silence is optional (.):
    cmd = f"{timit_norm_trans_script} -i {data}/stm -m {phonemap} -from 48 -to 39 | \
     sed 's: sil: (sil):g' > {dir}/scoring/stm_39phn"
    run_shell(cmd)

    copy(os.path.join(data, "glm"), os.path.join(dir, "scoring", "glm_39phn"))

    if stage <= 0:
        # Get the phone-sequence on the best-path:
        for LMWT in range(min_lmwt, max_lmwt):
            acoustic_scale = 1 / LMWT * mbr_scale
            cmd = f"{pl_cmd_script} JOB=1:{nj} {dir}/scoring/log/best_path.{LMWT}.JOB.log " + \
                  f"lattice-align-phones {model} \"ark:gunzip -c {dir}/lat.JOB.gz|\" ark:- \| " + \
                  f"lattice-to-ctm-conf --acoustic-scale={acoustic_scale:.8f} --lm-scale={mbr_scale} ark:- {dir}/scoring/{LMWT}.JOB.ctm || exit 1;"
            run_shell(cmd)
            run_shell(f"cat {dir}/scoring/{LMWT}.*.ctm | sort > {dir}/scoring/{LMWT}.ctm")
            run_shell(f"rm {dir}/scoring/{LMWT}.*.ctm")

    if stage <= 1:
        # Map ctm to 39 phone classes:
        cmd = f"{pl_cmd_script} LMWT={min_lmwt}:{max_lmwt} {dir}/scoring/log/map_ctm.LMWT.log " + \
              f"mkdir {dir}/score_LMWT ';' " + \
              f"cat {dir}/scoring/LMWT.ctm \| " + \
              f"{int2sym_script} -f 5 {symtab} \| " + \
              f"{timit_norm_trans_script} -i - -m {phonemap} -from 48 -to 39 '>' " + \
              f"{dir}/scoring/LMWT.ctm_39phn || exit 1"
        run_shell(cmd)

    # Score the set...
    cmd = f"{pl_cmd_script} LMWT={min_lmwt}:{max_lmwt} {dir}/scoring/log/map_ctm.LMWT.log " + \
          f"cp {dir}/scoring/stm_39phn {dir}/score_LMWT/stm_39phn '&&' cp {dir}/scoring/LMWT.ctm_39phn {dir}/score_LMWT/ctm_39phn '&&' " + \
          f"{hubscr} -p {hubdir} -V -l english -h hub5 -g {dir}/scoring/glm_39phn -r {dir}/score_LMWT/stm_39phn {dir}/score_LMWT/ctm_39phn || exit 1;"
    run_shell(cmd)
