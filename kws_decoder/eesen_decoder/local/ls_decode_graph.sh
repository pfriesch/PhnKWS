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

# This script compiles the ARPA-formatted language models into FSTs. Finally it composes the LM, lexicon
# and token FSTs together into the decoding graph. 

. ./path.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 <lang-dir> <arpa-lm-path> <out_dir> <lm-dir-tmp>"
  echo "e.g.: data/lang ...lm.arpa out_dir data/lm_tmp"
  exit 1
fi

langdir=$1
#lmdir=$2
arpa_lm=$2
out_dir=$3
tmpdir=$4
mkdir -p $tmpdir

# Adapted from the kaldi-trunk librespeech script local/format_lms.sh
#echo "Preparing language models for testing, may take some time ... "
#for lm_suffix in tgsmall tgmed ; do  ###tglarge fglarge ; do
#test=${langdir}_test_${lm_suffix}
mkdir -p $out_dir
cp ${langdir}/words.txt $out_dir || exit 1;

#echo "-----------------------------------------"
#echo "Working on " $lm_suffix;
#echo "-----------------------------------------"

cat $arpa_lm | \
 utils/find_arpa_oovs.pl $out_dir/words.txt  > $tmpdir/oovs.txt

cat $arpa_lm | \
  arpa2fst - | fstprint | \
  utils/remove_oovs.pl $tmpdir/oovs.txt | \
  utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$out_dir/words.txt \
    --osymbols=$out_dir/words.txt  --keep_isymbols=false --keep_osymbols=false | \
   fstrmepsilon | fstarcsort --sort_type=ilabel > $out_dir/G.fst

# Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
# minimized.
fsttablecompose ${langdir}/L.fst $out_dir/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst || exit 1;
fsttablecompose ${langdir}/T.fst $tmpdir/LG.fst > $out_dir/TLG.fst || exit 1;
#done

echo "Composing decoding graph TLG.fst succeeded"
#rm -r $tmpdir
