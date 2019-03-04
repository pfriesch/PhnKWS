import os

from tqdm import tqdm

from kaldi_decoding_scripts.local.just_transcript import get_transcripts
from utils.logger_config import logger
from utils.utils import run_shell



def decode_ctc(words_path,
               graph_path,
               out_folder,
               featstrings,
               min_active=20,
               max_active=700,
               max_mem=500000,
               beam=5.0,
               latbeam=5.0,
               acwt=1.0,
               **kwargs):
    out_folder = f"{out_folder}/exp_files"

    acwt = 1.0

    assert graph_path.endswith("TLG.fst")
    assert words_path.endswith("words.txt")

    assert isinstance(featstrings, list)

    assert os.environ['EESEN_ROOT']
    latgen_faster_bin = f"{os.environ['EESEN_ROOT']}/src/decoderbin/latgen-faster"

    chnk_id = 0
    for ck_data in tqdm(featstrings, desc="lattice generation chunk:"):
        finalfeats = f"ark,s,cs: "


        # Decode for each of the acoustic scales
        run_shell(f"{latgen_faster_bin} "
                  + f"--max-active={max_active} "
                  + f"--max-mem={max_mem} "
                  + f"--beam={beam} "
                  + f"--lattice-beam={latbeam} "
                  + f"--acoustic-scale={acwt} "
                  + f"--allow-partial=true "
                  + f"--word-symbol-table={words_path} "
                  + f"{graph_path} "
                  + f"ark:{ck_data} \"ark:|gzip -c > {out_folder}/lat.{chnk_id}.gz\"")


    transcripts_best, transcripts, lattice_confidence, lm_posterior, acoustic_posterior = get_transcripts(words_path,
                                                                                                          out_folder)

    for t in transcripts_best:
        assert transcripts_best[t] == transcripts[t], f"{t}: {transcripts_best[t]} =!= {transcripts[t]}"

    assert len(transcripts) == len(lattice_confidence)
    transcripts = dict(transcripts)
    lattice_confidence = dict(lattice_confidence)
    lm_posterior = dict(lm_posterior)
    acoustic_posterior = dict(acoustic_posterior)
    result = {}
    for sample_id in transcripts:
        _lattice_confidence = lattice_confidence[sample_id] \
            if 10000000000.0 != lattice_confidence[sample_id] else float("inf")
        _lm_posterior = lm_posterior[sample_id]  # TODO normalize
        _acoustic_posterior = acoustic_posterior[sample_id]  # TODO normalize
        result[sample_id] = (transcripts[sample_id], _lattice_confidence, _lm_posterior, _acoustic_posterior)

    return result
