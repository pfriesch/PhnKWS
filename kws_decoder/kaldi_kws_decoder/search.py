import os

# Begin configuration section.
from kws_decoder.kaldi_kws_decoder.make_index import make_index
from utils.utils import run_shell


def kws_search(langdir,  # ="data/lang",
               datadir,  # ="data/dev_clean_2",
               decodedir,  # ="exp/chain/tdnn1h_sp_online/decode_tglarge_dev_clean_2",

               min_lmwt=8,
               max_lmwt=12,
               cmd="run.pl",
               alignment_model=None,
               # skip_scoring=False,
               skip_optimization="false",
               max_states=350000,
               # indices_dir="exp/chain/tdnn1h_sp_online/decode_tglarge_dev_clean_2//kws_indices",
               # indices_dir=None,
               kwsout_dir=None,
               word_ins_penalty=0,
               # extraid=None,
               silence_word=None,
               strict="false",
               duptime=0.6,
               ntrue_scale=1.0,
               frame_subsampling_factor=1,
               nbest=-1,
               max_silence_frames=50,
               skip_indexing=True):
    kwsdatadir = f"{datadir}/kws"
    kwsoutdir = f"{decodedir}/kws"

    # if indices_dir is None:
    indices_dir = kwsoutdir

    if not os.path.isdir(kwsoutdir):
        os.makedirs(kwsoutdir)
    for d in [datadir, kwsdatadir, langdir, decodedir]:
        if not os.path.isdir(d):
            raise RuntimeError(f"FATAL: expected directory {d} to exist")

    # print(f"Searching: {kwsdatadir}")
    # with open(f"{kwsdatadir}/trials") as f:
    #     duration = float(f.readlines()[0].strip())
    # print(f"Duration: {duration}")

    frame_subsampling_factor = 1
    # if os.path.exists(f"{decodedir}/../frame_subsampling_factor"):
    #     with open(f"{decodedir}/../frame_subsampling_factor") as f:
    #         frame_subsampling_factor = int(f.readlines()[0].strip())
    #     print(f"Frame subsampling factor autodetected: {frame_subsampling_factor}")
    #
    # elif os.path.exists(f"{decodedir}/../../frame_subsampling_factor"):
    #     with open(f"{decodedir}/../../frame_subsampling_factor") as f:
    #         frame_subsampling_factor = int(f.readlines()[0].strip())
    #     print(f"Frame subsampling factor autodetected: {frame_subsampling_factor}")
    # else:
    #     print(f"No Frame subsampling factor autodetected!")

    # if not os.path.exists(f"{indices_dir}/.done.index") and not skip_indexing:
    if not os.path.isdir(indices_dir):
        os.makedirs(indices_dir)
    for lmwt in range(min_lmwt, max_lmwt):
        indices = f"{indices_dir}_{lmwt}"
        if not os.path.isdir(indices):
            os.makedirs(indices)

        acwt = 1.0 / lmwt

        make_index(kwsdatadir, langdir, decodedir, indices, alignment_model, acwt, lmwt, max_silence_frames, max_states,
                   word_ins_penalty=word_ins_penalty, skip_optimization=skip_optimization,
                   frame_subsampling_factor=1)

    # run_shell(f"touch {indices_dir}/.done.index")
    # else:
    #     print("Assuming indexing has been aready done. If you really need to re-run ")
    #     print(f"the indexing again, delete the file {indices_dir}/.done.index")

    keywords = f"{kwsdatadir}/keywords.fsts"
    if os.path.exists(keywords):
        print(f"Using {keywords} for search")
        keywords = f"ark:{keywords}"

    elif os.path.exists(f"{keywords}.gz"):
        print(f"Using {keywords}.gz for search")
        keywords = f"ark:gunzip -c {keywords}.gz |"
    else:
        print(f"The keyword file {keywords}.gz does not exist")

    for lmwt in range(min_lmwt, max_lmwt):
        kwsoutput = f"{kwsoutdir}_{lmwt}"
        indices = f"{indices_dir}_{lmwt}"

        _index_file = f"{indices}/index.1.gz"
        if os.path.exists(_index_file):
            print(f"no such file {_index_file}")

        if not os.path.isdir(f"{kwsoutput}/log"):
            os.makedirs(f"{kwsoutput}/log")

        run_shell(f"kws-search --strict={strict} --negative-tolerance=-1 " \
                  + f"--frame-subsampling-factor={frame_subsampling_factor} " \
                  + f"\"ark:gzip -cdf {indices}/index.1.gz |\" \"{keywords}\" " \
                  + f"\"ark,t:| sort -u | gzip -c > {kwsoutput}/result.1.gz\" " \
                  + f"\"ark,t:| sort -u | gzip -c > {kwsoutput}/stats.1.gz\" ")

    for lmwt in range(min_lmwt, max_lmwt):
        kwsoutput = f"{kwsoutdir}_{lmwt}"
        indices = f"{indices_dir}_{lmwt}"

        # This is a memory-efficient way how to do the filtration
        # we do this in this way because the result.* files can be fairly big
        # and we do not want to run into troubles with memory
        results = []
        if os.path.exists(f"{kwsoutput}/result.1.gz"):
            kws_res = run_shell(f"gunzip -c {kwsoutput}/result.1.gz")
            if len(kws_res) > 0:
                results.append(kws_res)
        elif os.path.exists(f"{kwsoutput}/result.1"):
            kws_res = run_shell(f"cat {kwsoutput}/result.1")
            if len(kws_res) > 0:
                results.append(kws_res)
        else:
            raise RuntimeError(f"The file {kwsoutput}/result.1[.gz] does not exist")
        # we have to call it using eval as we need the bash to interpret
        # the (possible) command substitution in case of gz files
        # bash -c would probably work as well, but would spawn another
        # shell instance

        results = results[0].split("\n")
        with open(f"{kwsoutput}/results", "w") as f:
            f.writelines(results)

    # if stage <= 4:
    #     if skip_scoring:
    #         print(f"Not scoring, because --skip-scoring true was issued")
    #     elif os.access(f"local/kws/score.sh", os.X_OK):
    #         print(f"Not scoring, because the file local/kws_score.sh is not present")
    #     else:
    #         print("Scoring KWS results")
    #         run_shell(f"local/kws/score.sh --cmd \"{cmd}\" " \
    #                   + f"--min-lmwt {min_lmwt} --max-lmwt {max_lmwt} {extraid_flags}" \
    #                   + f"{langdir} {datadir} {kwsoutdir}")

    print("Done")


if __name__ == '__main__':
    kws_search()
