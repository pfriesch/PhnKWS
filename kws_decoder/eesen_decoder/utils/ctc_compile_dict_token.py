#!/bin/bash
# Copyright 2015       Yajie Miao    (Carnegie Mellon University)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script compiles the lexicon and CTC tokens into FSTs. FST compiling slightly differs between the
# phoneme and character-based lexicons.
import os
from shutil import copy

dict_type = "phn"  # the type of lexicon, either "phn" or "char"
space_char = "<SPACE>"  # the character you have used to represent spaces

# if [ $# -ne 3 ]; then
#   echo "usage: utils/ctc_compile_dict_token.sh <dict-src-dir> <tmp-dir> <lang-dir>"
#   echo "e.g.: utils/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn"
#   echo "<dict-src-dir> should contain the following files:"
#   echo "lexicon.txt lexicon_numbers.txt units.txt"
#   echo "options: "
#   echo "     --dict-type <type of lexicon>                   # default: phn."
#   echo "     --space-char <space character>                  # default: <SPACE>, the character to represent spaces."
#   exit 1;
# fi


libri_kw2phn_mapping = {
    "backward": ["b ae1 k w er0 d"],
    "bed": ["b eh1 d"],
    "bird": ["b er1 d"],
    "cat": ["k ae1 t"],
    "dog": ["d ao1 g"],
    "down": ["d aw1 n"],
    "eight": ["ey1 t"],
    "five": ["f ay1 v"],
    "follow": ["f aa1 l ow0"],
    "forward": ["f ao1 r w er0 d"],
    "four": ["f ao1 r"],
    "go": ["g ow1"],
    "happy": ["hh ae1 p iy0"],
    "house": ["hh aw1 s"],
    "learn": ["l er1 n"],
    "left": ["ll eh1 f t"],
    "marvin": ["m aa1 r v ih0 n"],
    "nine": ["n ay1 n"],
    "no": ["n ow1"],
    "off": ["ao1 f"],
    "on": ["aa1 n", "ao1 n", "ow1 n"],
    "one": ["hh w ah1 n", "w ah1 n"],
    "right": ["r ay1 t"],
    "seven": ["s eh1 v ah0 n"],
    "sheila": ["sh iy1 l ah0"],
    "six": ["s ih1 k s"],
    "stop": ["s t aa1 p"],
    "three": ["th r iy1"],
    "tree": ["t r iy1"],
    "two": ["t uw1"],
    "up": ["ah1 p"],
    "visual": ["v ih1 zh ah0 w ah0 l"],
    "wow": ["w aw1"],
    "yes": ["y eh1 s", "y z"],
    "zero": ["z ih1 r ow0", "z iy1 r ow0"]
}


def ctc_compile_dict_token(srcdir="./exp/nml_seq_fw_seq_tw/dict", tmpdir="./exp/nml_seq_fw_seq_tw/lang/tmp",
                           _dir="./exp/nml_seq_fw_seq_tw/lang", dict_type="char", space_char="<SPACE>"):
    os.makedirs(_dir, exist_ok=True)
    os.makedirs(tmpdir, exist_ok=True)

    for _txt in ["lexicon_numbers.txt", "units.txt"]:
        copy(f"{srcdir}/{_txt}", dir)

    # # Add probabilities to lexicon entries. There is in fact no point of doing this here since all the entries have 1.0.
    # # But utils/make_lexicon_fst.pl requires a probabilistic version, so we just leave it as it is.
    #### perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < ./exp/nml_seq_fw_seq_tw/dict/keyword_lexicon_reduced.txt > ./exp/nml_seq_fw_seq_tw/lang/tmp/lexiconp.txt ;
    # EXAMPLE ( NO SIL ETC SUMBOLS
    #FOLLOW_0 1.0    F AA L OW
    # YES_0 1.0       Y EH S
    # YES_1 1.0       Y Z
    # FOUR_0 1.0      F AO R
    # OFF_0 1.0       AO F
    # ZERO_0 1.0      Z IH R OW
    # ZERO_1 1.0      Z IY R OW
    # NO_0 1.0        N OW
    # UP_0 1.0        AH P
    # WOW_0 1.0       W AW
    # SIX_0 1.0       S IH K S
    # TREE_0 1.0      T R IY
    #
    # # Add disambiguation symbols to the lexicon. This is necessary for determinizing the composition of L.fst and G.fst.
    # # Without these symbols, determinization will fail.
    #### ndisambig=`utils/add_lex_disambig.pl ./exp/nml_seq_fw_seq_tw/lang/tmp/lexiconp.txt ./exp/nml_seq_fw_seq_tw/lang/tmp/lexiconp_disambig.txt`
    #### ndisambig=$[$ndisambig+1];
    #
    #### ( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) > ./exp/nml_seq_fw_seq_tw/lang/tmp/disambig.list
    # EXAMPLE
    # #0
    # #1
    # #2
    #
    # # Get the full list of CTC tokens used in FST. These tokens include <eps>, the blank <blk>, the actual labels (e.g.,
    # # phonemes), and the disambiguation symbols.
    #### cat ./exp/nml_seq_fw_seq_tw/dict/units.txt | awk '{print $1}' > ./exp/nml_seq_fw_seq_tw/lang/tmp/units.list
    #EXAMPLE
    #<NOISE>
    # <SPOKEN_NOISE>
    # <SPACE>
    # <UNK>
    # AA
    # AE
    # AH
    # AO
    # AW
    #### (echo '<eps>'; echo '<blk>';) | cat - ./exp/nml_seq_fw_seq_tw/lang/tmp/units.list ./exp/nml_seq_fw_seq_tw/lang/tmp/disambig.list | awk '{print $1 " " (NR-1)}' > ./exp/nml_seq_fw_seq_tw/lang/tokens.txt
    #EXAMPLE
    #<eps> 0
    # <blk> 1
    # <NOISE> 2
    # <SPOKEN_NOISE> 3
    # <SPACE> 4
    # <UNK> 5
    # AA 6
    # AE 7
    # AH 8
    # AO 9
    # AW 10
    # AY 11
    # B 12
    # D 13
    # EH 14
    # ER 15
    # EY 16
    # F 17
    # G 18
    # HH 19
    # IH 20
    # IY 21
    # K 22
    # L 23
    # LL 24
    # M 25
    # N 26
    # OW 27
    # P 28
    # R 29
    # S 30
    # SH 31
    # T 32
    # TH 33
    # UW 34
    # V 35
    # W 36
    # Y 37
    # Z 38
    # ZH 39
    # #0 40
    # #1 41
    # #2 42
    # #!!!!!!!!!!!!! Compile the tokens into FST
    #### utils/ctc_token_fst.py ./exp/nml_seq_fw_seq_tw/lang/tokens.txt | fstcompile --isymbols=./exp/nml_seq_fw_seq_tw/lang/tokens.txt --osymbols=./exp/nml_seq_fw_seq_tw/lang/tokens.txt \
    ####    --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > ./exp/nml_seq_fw_seq_tw/lang/T.fst || exit 1;
    #
    # # Encode the words with indices. Will be used in lexicon and language model FST compiling.
    # cat ./exp/nml_seq_fw_seq_tw/lang/tmp/lexiconp.txt | awk '{print $1}' | sort | uniq  | awk '
    #   BEGIN {
    #     print "<eps> 0";
    #   }
    #   {
    #     printf("%s %d\n", $1, NR);
    #   }
    #   END {
    #     printf("#0 %d\n", NR+1);
    #   }' > ./exp/nml_seq_fw_seq_tw/lang/words.txt || exit 1;
    #
    # # Now compile the lexicon FST. Depending on the size of your lexicon, it may take some time.
    token_disambig_symbol=`grep \#0 ./exp/nml_seq_fw_seq_tw/lang/tokens.txt | awk '{print $2}'`
    word_disambig_symbol=`grep \#0 ./exp/nml_seq_fw_seq_tw/lang/words.txt | awk '{print $2}'`

    case phn in
      phn)
         utils/make_lexicon_fst.pl --pron-probs ./exp/nml_seq_fw_seq_tw/lang/tmp/lexiconp_disambig.txt 0 "sil" '#'$ndisambig | \
           fstcompile --isymbols=./exp/nml_seq_fw_seq_tw/lang/tokens.txt --osymbols=./exp/nml_seq_fw_seq_tw/lang/words.txt \
           --keep_isymbols=false --keep_osymbols=false |   \
           fstaddselfloops  "echo $token_disambig_symbol |" "echo $word_disambig_symbol |" | \
           fstarcsort --sort_type=olabel > ./exp/nml_seq_fw_seq_tw/lang/L.fst || exit 1;
           ;;
      char)
         echo "Building a character-based lexicon, with <SPACE> as the space"
         utils/make_lexicon_fst.pl --pron-probs ./exp/nml_seq_fw_seq_tw/lang/tmp/lexiconp_disambig_small.txt 0.5 "<SPACE>" '#'$ndisambig | \
           fstcompile --isymbols=./exp/nml_seq_fw_seq_tw/lang/tokens.txt --osymbols=./exp/nml_seq_fw_seq_tw/lang/words_small.txt \
           --keep_isymbols=false --keep_osymbols=false |   \
           fstaddselfloops  "echo $token_disambig_symbol |" "echo $word_disambig_symbol |" | \
           fstarcsort --sort_type=olabel > ./exp/nml_seq_fw_seq_tw/lang/L.fst || exit 1;
           ;;
      *) echo "$0: invalid dictionary type phn" && exit 1;
    esac

    echo "Dict and token FSTs compiling succeeded"
