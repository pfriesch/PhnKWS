import os
from glob import glob

from kaldi_decoding_scripts.utils.int2sym import int2sym
from utils.utils import run_shell


def get_transcripts(words_path, workdir):
    decoing_scripts_folder = os.path.join(os.getcwd(), __name__.split(".")[0])  # 'kaldi_decoding_scripts']
    int2sym_script = os.path.join(decoing_scripts_folder, "utils/int2sym.pl")

    assert len(glob(f"{workdir}/lat.*.gz")) > 0
    assert os.path.exists(int2sym_script)
    assert os.access(int2sym_script, os.X_OK)

    if not os.path.isdir(os.path.join(workdir, "scoring", "log")):
        os.makedirs(os.path.join(workdir, "scoring", "log"))

    for file in glob(f"{workdir}/lat.*.gz"):
        assert os.stat(file).st_size > 20, f"{file} seems to be empty with size of {os.stat(file).st_size} bytes"

    # TODO think about if each of these scalings and penalties make sense
    # TODO look into lattice-to-post --acoustic-scale=0.1 ark:1.lats ark:- | \
    #     gmm-acc-stats 10.mdl "$feats" ark:- 1.acc for confidence/sensitivity
    # TODO look into   lattice-to-fst --lm-scale=0.0 --acoustic-scale=0.0 ark:1.lats ark:1.words
    # to visualize the lattice and how the pruned fst looks like

    # cmd = f"lattice-scale --inv-acoustic-scale={language_model_weigth} " \
    #       + f"\"ark:gunzip -c {workdir}/lat.*.gz |\" ark:- | " \
    #       + f"lattice-add-penalty --word-ins-penalty={word_ins_penalty} ark:- ark:- | " \
    #       + f"lattice-best-path --word-symbol-table={words_path} ark:- " \
    #       + f"ark,t:{workdir}/scoring/{language_model_weigth}.{word_ins_penalty}.tra"
    # run_shell(cmd)

    ali_out_file = "/dev/null"
    transcript_out_file = f"{workdir}/scoring/keywords.tra"
    lm_posterior_out_file = f"{workdir}/scoring/keywords.lm_post"
    acoustic_posterior_out_file = f"{workdir}/scoring/keywords.ac_post"

    # plt =True
    # if plt:
    # only for word sequences, not useful for KWS
    #     run_shell(f"gunzip -c {workdir}/lat.*.gz | lattice-to-fst ark:- \"scp,p:echo $utterance_id $tmp_dir/$utterance_id.fst|\"")
        # run_shell(f"gunzip -c {workdir}/lat.*.gz | lattice-to-fst ark:- ")


    run_shell(f"gunzip -c {workdir}/lat.*.gz | "
              + f"lattice-to-nbest ark:- ark,t:- | "
              + f"nbest-to-linear ark:- ark:{ali_out_file} ark,t:{transcript_out_file} "
              + f"ark,t:{lm_posterior_out_file} ark,t:{acoustic_posterior_out_file}")
    transcripts = int2sym(transcript_out_file, words_path)
    with open(lm_posterior_out_file, "r") as f:
        lm_posterior = f.readlines()
    lm_posterior = [line.strip().split(" ") for line in lm_posterior if line != ""]
    lm_posterior = [(sample_id[:-2] if sample_id.endswith("-1") else sample_id,
                     float(posterior))
                    for sample_id, posterior in lm_posterior]

    with open(acoustic_posterior_out_file, "r") as f:
        acoustic_posterior = f.readlines()
    # acoustic_posterior = [line.split(" ") for line in acoustic_posterior.split("\n") if line != ""]

    acoustic_posterior = [line.strip().split(" ") for line in acoustic_posterior if line != ""]
    acoustic_posterior = [(sample_id[:-2] if sample_id.endswith("-1") else sample_id,
                           float(posterior))
                          for sample_id, posterior in acoustic_posterior]

    run_shell(f"gunzip -c {workdir}/lat.*.gz | "
              + f"lattice-best-path --word-symbol-table={words_path} ark:- "
              + f"ark,t:{workdir}/scoring/keywords_best.tra")
    transcripts_best = int2sym(f"{workdir}/scoring/keywords_best.tra",
                               words_path)

    lattice_confidence = run_shell(f"gunzip -c {workdir}/lat.*.gz | " \
                                   + f"lattice-confidence ark:- ark,t:-")
    lattice_confidence = [line.strip().split(" ", 1) for line in lattice_confidence.split("\n") if line != ""]
    lattice_confidence = [(sample_id, float(confidence)) for sample_id, confidence in lattice_confidence]

    # run_shell(f"cat {workdir}/scoring/keywords.tra")

    # run_shell(f"gunzip -c {workdir}/lat.*.gz | " \
    #           + f"lattice-1best ark:- ark,t:{workdir}/scoring/keywords.lat")

    # ali_model = '/mnt/data/pytorch-kaldi/tmp/graph_final/final.mdl'
    # run_shell(f"gunzip -c {workdir}/lat.*.gz | " \
    #           + f"lattice-to-nbest ark:- ark,t:- | nbest-to-linear ark:- ark,t:- | " +
    #           f" lattice-to-fst ark:- \"scp,p:echo tree_fc2411fe_nohash_2 /tmp/kaldi.UIEL/tree_fc2411fe_nohash_2.fst|\" ")

    # nbest-to-linear ark,t:1.ali 'ark,t:1.tra' ark,t:1.lm ark,t:1.ac

    # run_shell(f"fstdraw /tmp/kaldi.UIEL/tree_fc2411fe_nohash_2.fst")

    # run_shell(f"cat {workdir}/scoring/keywords.lat | " \
    #           + f" lattice-to-fst")

    # TODO lattice-confidence
    # Compute sentence-level lattice confidence measures for each lattice.
    # The output is simly the difference between the total costs of the best and
    # second-best paths in the lattice (or a very large value if the lattice
    # had only one path).  Caution: this is not necessarily a very good confidence
    # measure.  You almost certainly want to specify the acoustic scale.
    # If the input is a state-level lattice, you need to specify
    # --read-compact-lattice=false, or the confidences will be very small
    # (and wrong).  You can get word-level confidence info from lattice-mbr-decode.

    return transcripts_best, transcripts, lattice_confidence, lm_posterior, acoustic_posterior
