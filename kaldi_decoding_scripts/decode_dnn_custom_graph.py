from shutil import copy
import os
from glob import glob
import re

from tqdm import tqdm

from kaldi_decoding_scripts.local.just_transcript import get_transcripts
from utils.utils import run_shell
from utils.logger_config import logger


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
           scoring_type="std",  # none, std & basic so far
           scoring_opts=None,
           norm_vars=False,
           **kwargs):
    out_folder = f"{out_folder}/exp_files"
    # TODO remove
    if scoring_opts == '"--min-lmwt 1 --max-lmwt 10"':
        scoring_opts = {"min_lmwt": 1, "max_lmwt": 10}
    if scoring_opts is None:
        scoring_opts = {"min_lmwt": 1, "max_lmwt": 10}
    assert isinstance(featstrings, list)
    num_threads = 2  # TODO more threads
    assert out_folder[-1] != '/'
    srcdir = os.path.dirname(out_folder)

    thread_string = "-parallel --num-threads={}".format(num_threads)

    if not os.path.isdir(os.path.join(out_folder, "log")):
        os.makedirs(os.path.join(out_folder, "log"))

    num_jobs = len(featstrings)

    with open(os.path.join(out_folder, "num_jobs"), "w") as f:
        f.write(str(num_jobs))

    # assert os.path.exists(os.path.join(graphdir, "HCLG.fst"))
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
              + f'\"{finalfeats}\" \"ark:|gzip -c > {out_folder}/lat.{chnk_id}.gz\" &> {out_folder}/log/decode.{chnk_id}.log'
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


def best_wer(decoding_dir, scoring_type):
    if scoring_type != "none":
        if scoring_type == "std" or scoring_type == "basic":
            return best_wer_timit(decoding_dir)
        elif scoring_type == "libri":
            return best_wer_libri(decoding_dir)
        else:
            raise ValueError
    else:
        raise ValueError


def best_wer_libri(decoding_dir):
    avg_lines = []
    for path in glob(os.path.join(decoding_dir, "wer_*")):
        _, LMWT, word_ins_penalty = os.path.basename(path).split("_")
        with open(path, "r") as  f:
            wer_file_lines = f.readlines()
        avg_lines.append((LMWT, word_ins_penalty, next(line for line in wer_file_lines if "WER" in line)))

    result = []
    for line in avg_lines:
        _split = line[2].split(" ")
        wer = float(_split[1])
        total_fail = int(_split[3])
        total_possible = int(_split[5].strip(","))
        ins = int(_split[6])
        _del = int(_split[8])
        sub = int(_split[10])

        result.append({"lm_weight": int(line[0]), "word_ins_penalty": float(line[1]),
                       "wer": wer, "total": f"{total_fail}/{total_possible}",
                       "del": _del, "ins": ins, "sub": sub})

    return min(result, key=lambda x: x['wer'])


def best_wer_timit(decoding_dir):
    avg_lines = []
    for path in glob(os.path.join(decoding_dir, "score_*", "*.sys")):
        with open(path, "r") as  f:
            sys_lines = f.readlines()
        LMWT = os.path.basename(os.path.dirname(path)).split("_")[1]
        avg_lines.append((LMWT, next(line for line in sys_lines if "Sum/Avg" in line)))

    result = []
    for line in avg_lines:
        if line[1].count("|") == 5:
            _, _, n1, n2, n3, _ = line[1].split("|")

            _, corr, sub, _del, ins, per, _, _ = re.sub(' +', ' ', n2).split(" ")
            result.append({"lm_weight": line[0], "corr": corr, "sub": sub, "del": _del, "ins": ins, "per": per})
        else:
            logger.warn("Skipping line: {}".format(line[1]))

    return min(result, key=lambda x: x['per'])
