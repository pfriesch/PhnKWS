import os

from utils.utils import run_shell


def score(data, words_path, dir,
          word_ins_penalty=None,
          min_acwt=1,
          max_acwt=20,
          acwt_factor=0.05  # the scaling factor for the acoustic scale. The scaling factor for acoustic likelihoods
          # needs to be 0.5 ~1.0. However, the job submission script can only take integers as the
          # job marker. That's why we set the acwt to be integers (5 ~ 10), but scale them with 0.1
          # when they are actually used.
          ):
    if word_ins_penalty is None:
        word_ins_penalty = [0.0, 0.5, 1.0, 1.5, 2.0]
    # decoing_scripts_folder = os.path.join(os.getcwd(), __name__.split(".")[0])  # 'kaldi_decoding_scripts'
    # pl_cmd_script = os.path.join(decoing_scripts_folder, "utils/run.pl")
    # assert os.path.exists(pl_cmd_script)
    # assert os.access(pl_cmd_script, os.X_OK)
    # symtab = os.path.join(lang_or_graph, "words.txt")
    # assert os.path.exists(symtab)
    # assert os.path.exists(os.path.join(dir, "lat.1.gz"))
    # assert os.path.exists(os.path.join(data, "text"))
    # int2sym_script = os.path.join(decoing_scripts_folder, "utils/int2sym.pl")
    # assert os.path.exists(int2sym_script)
    # assert os.access(int2sym_script, os.X_OK)
    # if not os.path.isdir(os.path.join(dir, "scoring", "log")):
    #     os.makedirs(os.path.join(dir, "scoring", "log"))

    # --cmd "$decode_cmd" --nj 10 --beam 17.0 --lattice_beam 8.0 --max-active 5000 --acwt 0.9 \
    #                   --skip true --splice true --splice-opts "--left-context=1 --right-context=1" --skip-frames 3 --skip-offset 1 \
    #                                   ${lang_dir}_test_${lm_suffix} $exp_base/$test $train_dir/decode_${test}_${lm_suffix} || exit 1;

    # words_path = "wrds.txt"

    if not os.path.exists(f"{dir}/scoring"):
        os.makedirs(f"{dir}/scoring")

    assert os.environ['EESEN_ROOT']
    lattice_scale_bin = f"{os.environ['EESEN_ROOT']}/src/decoderbin/lattice-scale"
    lattice_add_penalty_bin = f"{os.environ['EESEN_ROOT']}/src/decoderbin/lattice-add-penalty"
    lattice_best_path_bin = f"{os.environ['EESEN_ROOT']}/src/decoderbin/lattice-best-path"

    # for wip in word_ins_penalty:
    #     for ACWT in range(min_acwt, max_acwt):
    #         run_shell(
    #             f"{lattice_scale_bin} --acoustic-scale={ACWT} --ascale-factor={acwt_factor}  \"ark:gunzip -c {dir}/lat.*.gz|\" ark:- | "
    #             + f"{lattice_add_penalty_bin} --word-ins-penalty={wip} ark:- ark:- |"
    #             + f"{lattice_best_path_bin} --word-symbol-table={words_path} ark:- ark,t:{dir}/scoring/{ACWT}_{wip}_tra")

    # run_shell(f"cat {data}/text | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' > {dir}/scoring/test_filt.txt")
    run_shell(
        f"cat {data}/text | sed 's:<UNK>::g' | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' > {dir}/scoring/text_filt")

    compute_wer_bin = f"{os.environ['EESEN_ROOT']}/src/decoderbin/compute-wer"
    lattice_1best_bin = f"{os.environ['EESEN_ROOT']}/src/decoderbin/lattice-1best"
    nbest_to_ctm_bin = f"{os.environ['EESEN_ROOT']}/src/decoderbin/nbest-to-ctm"
    compute_wer_bin = f"{os.environ['EESEN_ROOT']}/src/decoderbin/compute-wer"

    int2sym_script = os.path.join(os.getcwd(), "kaldi_decoding_scripts/utils/int2sym.pl")
    assert os.path.exists(int2sym_script)

    # for wip in word_ins_penalty:
    #     for ACWT in range(min_acwt, max_acwt):
    #         run_shell(f"cat {dir}/scoring/{ACWT}_{wip}_tra | {int2sym_script} -f 2- {words_path} | "
    #                   + f" sed 's:<UNK>::g' | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' |"
    #                   + f"{compute_wer_bin} --text --mode=present ark:{dir}/scoring/text_filt  ark,p:- {dir}/details_{ACWT}_{wip} >& {dir}/wer_{ACWT}_{wip}")

    convert_ctm_script = os.path.join(os.getcwd(), "kws_decoder/eesen_utils/convert_ctm.pl")
    assert os.path.exists(convert_ctm_script)
    name = "test_name_"
    # for wip in word_ins_penalty:
    for ACWT in range(min_acwt, max_acwt):
        if not os.path.exists(f"{dir}/score_{ACWT}/"):
            os.makedirs(f"{dir}/score_{ACWT}/")

        run_shell(
            f"{lattice_1best_bin} --acoustic-scale={ACWT} --ascale-factor={acwt_factor} \"ark:gunzip -c {dir}/lat.*.gz|\" ark:- | "
            + f"{nbest_to_ctm_bin} ark:- - | "
            + f"{int2sym_script} -f 5 {words_path}  | "
            + f"{convert_ctm_script} {data}/segments {data}/reco2file_and_channel")

        run_shell(
            f"{lattice_1best_bin} --acoustic-scale={ACWT} --ascale-factor={acwt_factor} \"ark:gunzip -c {dir}/lat.*.gz|\" ark:- | "
            + f"{nbest_to_ctm_bin} ark:- - | "
            + f"{int2sym_script} -f 5 {words_path}  | "
            + f"{convert_ctm_script} {data}/segments {data}/reco2file_and_channel "
            + f"> {dir}/score_{ACWT}/{name}.ctm")
