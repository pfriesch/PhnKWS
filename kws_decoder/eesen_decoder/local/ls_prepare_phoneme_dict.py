#!/bin/bash

# Copyright 2015  Yajie Miao   (Carnegie Mellon University)
#           2017  Jayadev Billa (librispeech adaptations)

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

# Creates a lexicon in which each word is represented by the sequence of its characters (spelling). In theory, we can build such
# a lexicon from the words in the training transcripts.  However, for  consistent comparison to the phoneme-based system, we use
# the word list from the CMU dictionary. No pronunciations are employed.


# echo "Usage: $0 <lm-dir> <dict-dir> <dict-name>"
# echo "e.g.: data/lm data/dict librispeech-lexicon.txt"
import os

from asr_egs.librispeech.sh_utils import run_shell


def prepare_phoneme_dict(lm_dir='./exp/nml_seq_fw_seq_tw',
                         dst_dir='./exp/nml_seq_fw_seq_tw/dict',
                         dict_name='librispeech_phn_reduced_dict.txt'):
    # run this from ../
    phndir = "data/local/dict_phn"
    dir = "data/local/dict_char"
    # mkdir -p $dir

    os.makedirs(lm_dir, exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)

    # exists and has a size greater than zero
    assert os.path.exists(f"{lm_dir}/{dict_name}") and os.path.getsize(f"{lm_dir}/{dict_name}") > 0

    print("test")
    #TODO assert no stress marks in in phonemes
    #TODO assert unique words

    # For phoneme based speech recognition we provide the pronunciation so no need to expand to char
    # We keep only one pronunciation for each word. Other alternative pronunciations are discarded.
    # run_shell(f"cat {lm_dir}/{dict_name} | " + \
    #           """perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}'""" + \
    #           f"> {dst_dir}/keyword_lexicon_reduced.txt || exit 1;")


    # #  Get the set of lexicon units without noises
    # run_shell(
    ##### perl -nae 'shift @F; foreach my $w (@F) {print $w . "\n"}' ./exp/nml_seq_fw_seq_tw/dict/keyword_lexicon_reduced.txt | sort -u > ./exp/nml_seq_fw_seq_tw/dict/units_nosil.txt
    #TODO all nosil units AA AE AH ...
    #
    # #TODO Add special noises words & characters into the lexicon.  To be consistent with the blank <blk>,
    # # we add "< >" to the noises characters
    #### (echo '<SPOKEN_NOISE> <SPOKEN_NOISE>'; echo '<UNK> <UNK>'; echo '<NOISE> <NOISE>'; echo '<SPACE> <SPACE>';) | \
    #### cat - ./exp/nml_seq_fw_seq_tw/dict/keyword_lexicon_reduced.txt | sort | uniq > ./exp/nml_seq_fw_seq_tw/dict/lexicon.txt || exit 1;
    # EXAMPLE:
    #NO_0 N OW
    # <NOISE> <NOISE>
    # OFF_0 AO F
    # ON_0 AA N
    # ON_1 AO N
    # ON_2 OW N
    # ONE_0 HH W AH N
    # ONE_1 W AH N
    # RIGHT_0 R AY T
    # SEVEN_0 S EH V AH N
    # SHEILA_0 SH IY L AH
    # SIX_0 S IH K S
    # <SPACE> <SPACE>
    # <SPOKEN_NOISE> <SPOKEN_NOISE>
    # STOP_0 S T AA P
    # THREE_0 TH R IY
    # TREE_0 T R IY

    #
    # #TODO  The complete set of lexicon units, indexed by numbers starting from 1
    #### (echo '<NOISE>'; echo '<SPOKEN_NOISE>'; echo '<SPACE>'; echo '<UNK>'; ) | cat - ./exp/nml_seq_fw_seq_tw/dict/units_nosil.txt | awk '{print $1 " " NR}' > ./exp/nml_seq_fw_seq_tw/dict/units.txt
    #EXAMPLE
    #<NOISE> 1
    # <SPOKEN_NOISE> 2
    # <SPACE> 3
    # <UNK> 4
    # AA 5
    # AE 6
    # AH 7
    # AO 8

    # #TODO Convert character sequences into the corresponding sequences of units indices, encoded by units.txt
    #### utils/sym2int.pl -f 2- ./exp/nml_seq_fw_seq_tw/dict/units.txt < ./exp/nml_seq_fw_seq_tw/dict/lexicon.txt > ./exp/nml_seq_fw_seq_tw/dict/lexicon_numbers.txt
    #EXAMPLE
    #MARVIN_0 24 5 28 34 19 25
    # NINE_0 25 10 25
    # NO_0 25 26
    # <NOISE> 1
    # OFF_0 8 16
    # ON_0 5 25
    # ON_1 8 25
    # ON_2 26 25
    # ONE_0 18 35 7 25
    # ONE_1 35 7 25
    # RIGHT_0 28 10 31
    # SEVEN_0 29 13 34 7 25
    # SHEILA_0 30 20 22 7
    # SIX_0 29 19 21 29
    # <SPACE> 3
    # <SPOKEN_NOISE> 2
    # STOP_0 29 31 5 27
    # THREE_0 32 28 20
    # TREE_0 31 28 20


    # run_shell("""echo "Phoneme-based dictionary preparation succeeded""")
