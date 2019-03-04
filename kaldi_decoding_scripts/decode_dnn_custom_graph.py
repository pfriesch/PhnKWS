from shutil import copy
import os
from glob import glob

from tqdm import tqdm

from kaldi_decoding_scripts.local.just_transcript import get_transcripts
from utils.utils import run_shell


def decode(alignment_model_path,
           words_path,
           graph_path,
           out_folder,
           featstrings,
           min_active=20,
           max_active=700,
           max_mem=500000,
           beam=5.0,
           latbeam=5.0,
           acwt=1.0,
           max_arcs=-1.0,
           **kwargs):
    out_folder = f"{out_folder}/exp_files"
    assert isinstance(featstrings, list)
    num_threads = 4  # TODO more threads
    assert out_folder[-1] != '/'
    srcdir = os.path.dirname(out_folder)

    thread_string = f"-parallel --num-threads={num_threads}"

    if not os.path.isdir(os.path.join(out_folder, "log")):
        os.makedirs(os.path.join(out_folder, "log"))

    assert graph_path.endswith("HCLG.fst")
    assert words_path.endswith("words.txt")
    assert alignment_model_path.endswith("final.mdl")

    # TODO should we really just delete these files?
    if len(glob(f"{out_folder}/lat.*.gz")) > 0:
        for file in glob(f"{out_folder}/lat.*.gz"):
            os.remove(file)
    if len(glob(f"{out_folder}/log/decode.*.log")) > 0:
        for file in glob(f"{out_folder}/log/decode.*.log"):
            os.remove(file)

    chnk_id = 0
    for ck_data in tqdm(featstrings, desc="lattice generation chunk:"):
        assert not os.path.exists(f"{out_folder}/lat.{chnk_id}.gz")
        assert not os.path.exists(f"{out_folder}/log/decode.{chnk_id}.log")
        finalfeats = f"ark,s,cs: cat {ck_data} |"
        cmd = f'latgen-faster-mapped{thread_string} --min-active={min_active} ' \
              + f'--max-active={max_active} --max-mem={max_mem} ' \
              + f'--beam={beam} --lattice-beam={latbeam} ' \
              + f'--acoustic-scale={acwt}' \
              + f' --allow-partial=true ' \
              + f'--word-symbol-table={words_path} {alignment_model_path} ' \
              + f'{graph_path} ' \
              + f'\"{finalfeats}\" \"ark:|gzip -c > {out_folder}/lat.{chnk_id}.gz\"'
        run_shell(cmd)
        chnk_id += 1

        # TODO display the generated lattice for keywords

    copy(alignment_model_path, srcdir)
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
