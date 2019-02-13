import os

from utils.logger_config import logger
from utils.utils import run_shell
from kws_decoder.arpa_utils import make_kw_arpa, UNK_WORD


def make_ctc_decoding_graph(keywords=['alexa', "left", "visual", "marvin"]):
    assert os.getcwd().endswith("eesen_decoder")  # TODO handle the relative folders
    eesen_utils = "utils"
    eesen_local = "local"

    tmpdir = "tmp"
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)
    final_lang_dir = "lang"
    if not hasattr(logger, "logger"):
        logger.configure_logger(tmpdir)

    dict_dir = "in_tmp"
    if not os.path.isdir(dict_dir):
        os.makedirs(dict_dir)

    libri_lexicon = f"{tmpdir}/librispeech-lexicon.txt"
    if not os.path.exists(libri_lexicon):
        run_shell(f"wget -P {tmpdir}/ http://www.openslr.org/resources/11/librispeech-lexicon.txt")

    keywords = [kw.upper() for kw in keywords]

    keyword_pron = {}
    lines = []
    # lines = ["!SIL SIL\n", #TODO
    #          "<SPOKEN_NOISE> SPN\n",
    #          f"{UNK_WORD} SPN\n"]

    with open(libri_lexicon, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                _line = line.strip().split("\t", maxsplit=1)
                word = _line[0]
                pronounceiation = _line[1]
            else:
                _line = line.strip().split(" ", maxsplit=1)
                word = _line[0]
                pronounceiation = _line[1]

            if word in keywords:
                assert word not in keyword_pron
                keyword_pron[word] = pronounceiation.strip()
                lines.append(line)

    lexicon_name = "lexicon.txt"
    with open(f"{dict_dir}/{lexicon_name}", "w", encoding="utf-8") as f:
        f.writelines(sorted(lines))

    lexicon_name_reduced = "lexicon_reduced.txt"

    run_shell(f"perl -nae '$w = shift @F; if(! $seen{{$w}}) {{$seen{{$w}} = 1; $prn=join(\" \", @F); "
              + f"$prn =~ s/(\D)\d(\s*)/$1$2/g; print \"$w $prn\n\";}}' {dict_dir}/{lexicon_name} > {dict_dir}/{lexicon_name_reduced}")

    run_shell(f"{eesen_local}/ls_prepare_phoneme_dict.sh {dict_dir} {dict_dir} {lexicon_name_reduced}")
    dict_type = "phn"
    run_shell(
        f"{eesen_utils}/ctc_compile_dict_token.sh --dict-type {dict_type} {dict_dir} {tmpdir} {final_lang_dir} ")

    arpa_str = make_kw_arpa(keywords)
    arpa_lm = f"{final_lang_dir}/keyword.arpa"
    with open(arpa_lm, "w", encoding="utf-8") as f:
        f.writelines(arpa_str)

    #
    # unk_fst_dir = "unk_fst"
    #
    # #####Optional  UNK FST
    # # in librispeech_workdir
    # # or reduce num ngram option
    # ## using bigram only and num-ngrams is only 3336
    # # num_extra_ngrams=1000
    # run_shell(f"utils/lang/make_unk_lm.sh --num_extra_ngrams 1000 --ngram-order 2 {dict_dir} {unk_fst_dir}")

    # TODO alternative simple phone loop

    # prepare_lang_script = f"utils/prepare_lang.sh"
    # run_shell(f"{prepare_lang_script} --unk-fst {unk_fst_dir}/unk_fst.txt {dict_dir} \"<UNK>\" {tmpdir} {final_lang_dir}")

    final_graph_dir = "graph_final"
    run_shell(f"{eesen_local}/ls_decode_graph.sh {final_lang_dir} {arpa_lm} {final_graph_dir} {tmpdir}")

    return os.path.abspath(final_graph_dir)


if __name__ == '__main__':
    print((make_ctc_decoding_graph()))
