from shutil import copy
import os
from glob import glob
import re

from kaldi_decoding_scripts.local.score import score as score
from kaldi_decoding_scripts.local.score_basic import score as score_basic
from kaldi_decoding_scripts.local.score_libri import score as score_libri
from utils.utils import run_shell
from utils.logger_config import logger


def decode(alidir,
           data,
           graphdir,
           out_folder,
           featstrings,
           min_active=200,
           max_active=7000,
           max_mem=50000000,
           beam=13.0,
           latbeam=8.0,
           acwt=0.2,
           max_arcs=-1.0,
           scoring_type="std",  # none, std & basic so far
           scoring_opts=None,
           norm_vars=False,
           **kwargs):
    if scoring_opts is None:
        scoring_opts = {"min-lmwt": 1, "max-lmwt": 10}
    assert isinstance(featstrings, list)
    num_threads = 1
    assert out_folder[-1] != '/'
    srcdir = os.path.dirname(out_folder)

    thread_string = "-parallel --num-threads={}".format(num_threads)

    os.makedirs(os.path.join(out_folder, "log"))

    num_jobs = len(featstrings)

    with open(os.path.join(out_folder, "num_jobs"), "w") as f:
        f.write(str(num_jobs))

    assert os.path.exists(os.path.join(graphdir, "HCLG.fst"))

    JOB = 1
    for ck_data in featstrings:
        finalfeats = f"ark,s,cs: cat {ck_data} |"
        cmd = f'latgen-faster-mapped{thread_string} --min-active={min_active} ' + \
              f'--max-active={max_active} --max-mem={max_mem} ' + \
              f'--beam={beam} --lattice-beam={latbeam} ' + \
              f'--acoustic-scale={acwt} --allow-partial=true ' + \
              f'--word-symbol-table={graphdir}/words.txt {alidir}/final.mdl ' + \
              f'{graphdir}/HCLG.fst ' + \
              f'\"{finalfeats}\" \"ark:|gzip -c > {out_folder}/lat.{JOB}.gz\" &> {out_folder}/log/decode.{JOB}.log &'
        run_shell(cmd)
        JOB += 1

    copy(os.path.join(alidir, "final.mdl"), srcdir)

    if scoring_type != "none":
        if scoring_type == "std":
            score(data, graphdir, out_folder, num_jobs, **scoring_opts)
        elif scoring_type == "basic":
            score_basic(data, graphdir, out_folder, num_jobs, **scoring_opts)
        elif scoring_type == "libri":
            score_libri(data, graphdir, out_folder, **scoring_opts)
        else:
            raise ValueError


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
