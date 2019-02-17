import os

from utils.utils import run_shell


def make_index(kwsdatadir,
               langdir,
               decodedir,
               kwsdir,

               model=None, acwt=0.083333, lmwt=1.0, max_silence_frames=50, max_states=1000000, max_states_scale=4,
               max_expand=180,
               strict=True,
               word_ins_penalty=0, skip_optimization=False, frame_subsampling_factor=1

               ):
    """

    :param model: which model to use, speaker-adapted decoding"
    :param acwt: acoustic scale used for lattice
    :param lmwt: lm scale used for lattice
    :param max_silence_frames: maximum #frames for silence
    :param max_states:
    :param max_states_scale:
    :param max_expand: limit memory blowup in lattice-align-words
    :param strict:
    :param word_ins_penalty:
    :param silence_word: Specify this only if you did so in kws_setup
    :param skip_optimization:   If you only search for few thousands of keywords, you probablly
                                can skip the optimization; but if you're going to search for
                                millions of keywords, you'd better do set this optimization to
                                false and do the optimization on the final index.
    :param frame_subsampling_factor:    We will try to autodetect this. You should specify
                                        the right value if your directory structure is
                                        non-standard
    :return:
    """
    # The model directory is one level up from decoding directory.
    srcdir = os.path.dirname(decodedir)

    utter_id = f"{kwsdatadir}/utter_id"
    if not os.path.exists(utter_id):
        utter_id = f"{kwsdatadir}/utt.map"

    # if --model <mdl> was not specified on the command line...
    if model is None:
        model = f"{srcdir}/final.mdl"

    for f in [model, f"{decodedir}/lat.1.gz", utter_id]:
        if not os.path.exists(f):
            raise RuntimeError(f"Error: no such file {f}")

    # TODO assert silence word is !SIL or SIL and id is 1 (mostly id is 1)
    silence_int = 1
    silence_opt = f"--silence-label={silence_int}"

    # word_boundary = f"{langdir}/phones/word_boundary.int"
    word_boundary = f"/mnt/data/libs/kaldi/egs/librispeech/s5/data/lang/phones/word_boundary.int"

    align_lexicon = f"{langdir}/phones/align_lexicon.int"
    # if os.path.exists(word_boundary):
    run_shell(
        f"lattice-add-penalty --word-ins-penalty={word_ins_penalty} \"ark:gzip -cdf {decodedir}/lat.1.gz|\" ark:-  | " \
        # lattice-align-words {silence_opt} --max-expand={max_expand} {word_boundary} {model}  ark:- ark:-  | \
        + f"lattice-scale --acoustic-scale={acwt} --lm-scale={lmwt} ark:- ark:-  | \
      lattice-to-kws-index --max-states-scale={max_states_scale} --allow-partial=true \
      --frame-subsampling-factor={frame_subsampling_factor} \
      --max-silence-frames={max_silence_frames} --strict={strict} ark:{utter_id} ark:- ark:-  | \
      kws-index-union --skip-optimization={skip_optimization} --strict={strict} --max-states={max_states} \
      ark:- \"ark:|gzip -c > {kwsdir}/index.1.gz\" ")

    # elif os.path.exists(align_lexicon):
    #
    #     run_shell(f"lattice-add-penalty --word-ins-penalty={word_ins_penalty} \"ark:gzip -cdf {decodedir}/lat.1.gz|\" ark:-  | \
    #       lattice-align-words-lexicon {silence_opt} --max-expand={max_expand} {align_lexicon} {model}  ark:- ark:-  | \
    #       lattice-scale --acoustic-scale={acwt} --lm-scale={lmwt} ark:- ark:-  | \
    #       lattice-to-kws-index --max-states-scale={max_states_scale} --allow-partial=true \
    #       --frame-subsampling-factor={frame_subsampling_factor} \
    #       --max-silence-frames={max_silence_frames} --strict={strict} ark:{utter_id} ark:- ark:-  | \
    #       kws-index-union --skip-optimization={skip_optimization} --strict={strict} --max-states={max_states} \
    #       ark:- \"ark:|gzip -c > {kwsdir}/index.1.gz\" ")
    # else:
    #     raise RuntimeError(
    #         f"Error: cannot find either word-boundary file {word_boundary} or alignment lexicon {align_lexicon}")
