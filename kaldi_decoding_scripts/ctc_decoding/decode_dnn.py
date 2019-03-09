import os
from glob import glob

from tqdm import tqdm

from kaldi_decoding_scripts.ctc_decoding.score import score
from utils.utils import run_shell


def decode_ctc(data,
               graphdir,
               out_folder,
               featstrings,
               min_active=20,
               max_active=5000,
               max_mem=500000,
               beam=17.0,
               latbeam=8.0,
               acwt=0.9,
               **kwargs):
    out_folder = f"{out_folder}/exp_files"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    assert os.path.exists(f"{graphdir}/TLG.fst")
    assert os.path.exists(f"{graphdir}/words.txt")

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
                  + f"--word-symbol-table={graphdir}/words.txt "
                  + f"{graphdir}/TLG.fst "
                  + f"ark:{ck_data} \"ark:|gzip -c > {out_folder}/lat.{chnk_id}.gz\"")

        score(data, f"{graphdir}/words.txt", out_folder)

# def best_wer_libri(decoding_dir):
#     avg_lines = []
#     for path in glob(os.path.join(decoding_dir, "wer_*")):
#         _, LMWT, word_ins_penalty = os.path.basename(path).split("_")
#         with open(path, "r") as  f:
#             wer_file_lines = f.readlines()
#         avg_lines.append((LMWT, word_ins_penalty, next(line for line in wer_file_lines if "WER" in line)))
#
#     result = []
#     for line in avg_lines:
#         _split = line[2].split(" ")
#         wer = float(_split[1])
#         total_fail = int(_split[3])
#         total_possible = int(_split[5].strip(","))
#         ins = int(_split[6])
#         _del = int(_split[8])
#         sub = int(_split[10])
#
#         result.append({"lm_weight": int(line[0]), "word_ins_penalty": float(line[1]),
#                        "wer": wer, "total": f"{total_fail}/{total_possible}",
#                        "del": _del, "ins": ins, "sub": sub})
#
#     return min(result, key=lambda x: x['wer'])
