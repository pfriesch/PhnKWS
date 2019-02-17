#!/bin/bash
# Copyright (c) 2018, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>)
# License: Apache 2.0
import itertools
import os
import shutil

from kws_decoder.kaldi_kws_decoder.search import kws_search
from utils.logger_config import logger
from utils.utils import run_shell, check_environment


def kws(alignment_model):
    # Begin configuration section.
    flen = 0.01
    stage = 0
    cmd = "run.pl"
    data = "data/dev_clean_2_2"
    lang = "data/lang"
    keywords = "local/kws/example/keywords.txt"
    output = "data/dev_clean_2_2/kws/"
    # End configuration section

    if not os.path.isdir(output):
        os.makedirs(output)
    logger.configure_logger(output)
    check_environment()

    ## generate the auxiliary data files
    ## utt.map
    ## wav.map
    ## trials
    ## frame_length
    ## keywords.int

    ## For simplicity, we do not generate the following files
    ## categories

    ## We will generate the following files later
    ## hitlist
    ## keywords.fsts

    # if not os.path.exists(f"{data}/utt2dur"):
    #     run_shell(f"utils/data/get_utt2dur.sh {data}")

    # TODO    duration=$(cat $data/utt2dur | awk '{sum += $2} END{print sum}' )

    print("")

    # echo $duration > {output}/trials
    # echo $flen > {output}/frame_length

    # echo "Number of trials: $(cat {output}/trials)"
    # echo "Frame lengths: $(cat {output}/frame_length)"

    # echo "Generating map files"
    # with open(f"{data}/utt2dur", "r") as f:
    #     utt2dur = f.readlines()
    # utt_map = enumerate([l.split(" ", 1)[0] for l in utt2dur])
    # with open(f"{data}/wav.scp", "r") as f:
    #     wav_scp = f.readlines()
    # wav_map = enumerate([l.split(" ", 1)[0] for l in wav_scp])

    # run_shell(f" cat {data}/utt2dur | awk 'BEGIN{{i=1}}; {{print $1, i; i+=1;}}' > {output}/utt.map")
    # cat $data/wav.scp | awk 'BEGIN{i=1}; {print $1, i; i+=1;}' > {output}/wav.map

    # shutil.copy(f"{lang}/words.txt", f"{output}/words.txt")
    # shutil.copy(keywords, f"{output}/keywords.txt")

    with open(keywords, "r") as f:
        kw_list = f.readlines()
    kw_list = [kw.strip().split(" ", 1) for kw in kw_list if kw.strip() != ""]
    all_keywords = list(zip(*kw_list))[1]

    unique_words = list(itertools.chain.from_iterable([kw.split(" ") for kw in all_keywords]))

    word_id_map = {word: word_id for word_id, word in enumerate(unique_words)}

    kw_list_ids = [(kw_id,
                    " ".join([str(word_id_map[_kw_split]) for _kw_split in kw.split(" ")]))
                   for kw_id, kw in kw_list]

    with open(f"{output}/keywords.int", "w") as f:
        f.writelines([kw_id + " " + word_ids + "\n" for kw_id, word_ids in kw_list_ids])

    #
    # kw2word_id = run_shell(
    #     f"cat {output}/keywords.txt | local/kws/keywords_to_indices.pl --map-oov 0  {output}/words.txt").split("\n")
    # kw2word_id = [kw for kw in set(kw2word_id) if kw != ""]
    # " | sort -u > {output}/keywords.int"

    # if [ $stage -le 3]; then
    ## this steps generates the file keywords.fsts

    ## compile the keywords (it's done via tmp work dirs, so that
    ## you can use the keywords filtering and then just run fsts-union
    run_shell(f"local/kws/compile_keywords.sh {output} {lang} {output}/tmp.2")
    run_shell(f"cp {output}/tmp.2/keywords.fsts {output}/keywords.fsts")

    # kws_search(lang, data, decodedir=f"{data}/decodedir", alignment_model=alignment_model)
    kws_search(lang, data, decodedir=f"exp/chain/tdnn1h_sp_online/decode_tglarge_dev_clean_2",
               alignment_model=alignment_model)


# for example
#    f"fsts-union scp:<(sort data/{_dir}/kwset_${set}/tmp*/keywords.scp) \/
#      ark,t:"|gzip -c >data/{_dir}/kwset_${set}/keywords.fsts.gz"
##
# fi


#
# system=exp/chain/tdnn1h_sp_online/decode_tglarge_dev_clean_2/
# if [ $stage -le 4 ]; then
#   ## this is not exactly necessary for a single system and single keyword set
#   ## but if you have multiple keyword sets, then it avoids having to recompute
#   ## the indices unnecesarily every time (see --indices-dir and --skip-indexing
#   ## parameters to the search script bellow).
#   for lmwt in `seq 8 14` ; do
#     steps/make_index.sh --cmd "$cmd" --lmwt $lmwt --acwt 1.0 \
#       --frame-subsampling-factor 3\
#       {output} {lang} $system $system/kws_indices_$lmwt
#   done
# fi
#
# if [ $stage -le 5 ]; then

# echo cmd "$cmd"
# echo indices-dir $system/kws_indices
# echo skip-indexing true
# echo lang {lang}
# echo data $data
# echo system $system
## find the hits, normalize and score

# local/kws/search.sh --cmd "$cmd" --min-lmwt 8 --max-lmwt 14  \
#   {lang} $data $system
# fi

# echo "Done"


if __name__ == '__main__':
    ali_model = "/mnt/data/libs/kaldi/egs/librispeech/s5/exp/tri4b/final.mdl"
    os.chdir("/mnt/data/libs/kaldi/egs/mini_librispeech/s5")
    kws(ali_model)
