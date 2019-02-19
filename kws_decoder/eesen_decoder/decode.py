import os

from utils.utils import run_shell
from utils.logger_config import logger


def main(graphdir, data, _dir,
         nj=16,
         acwt=0.9,
         # min_active=200,

         max_active=7000,  # max-active
         beam=15.0,  # beam used
         lattice_beam=8.0,
         max_mem=50000000,  # approx. limit to memory consumption during minimization in bytes

         norm_vars=None,
         add_deltas=None,
         splice=False,
         skip=False,
         splice_opts=None,
         skip_frames=None,
         skip_offset=0):
    assert not _dir.endswith("/")
    srcdir = os.path.dirname(_dir)

    # Check if necessary files exist.
    for f in [f"{graphdir}/TLG.fst", f"{srcdir}/label.counts", f"{data}/feats.scp"]:
        assert os.path.exists(f)

    if add_deltas is None:
        add_deltas = f"cat {srcdir}/add_deltas 2>/dev/null"
    if norm_vars is None:
        norm_vars = f"cat {srcdir}/norm_vars 2>/dev/null"

    sdata = f"{data}/split{nj}"

    ### Set up the features
    logger.debug(f"feature: norm_vars({norm_vars}) add_deltas({add_deltas})")
    feats = f"ark,s,cs:apply-cmvn --norm-vars={norm_vars} --utt2spk=ark:{sdata}/JOB/utt2spk " \
            + f"scp:{sdata}/JOB/cmvn.scp scp:{sdata}/JOB/feats.scp ark:- |"
    ##
    if splice:
        feats = f"{feats} splice-feats {splice_opts} ark:- ark:- |"
    if add_deltas:
        feats = f"{feats} add-deltas ark:- ark:- |"

    if skip:
        feats = f"{feats} subsample-feats --n={skip_frames} --offset={skip_offset} ark:- ark:- |"

    # Decode for each of the acoustic scales
    run_shell(f"net-output-extract --class-frame-counts={srcdir}/label.counts --apply-log=true {srcdir}/final.nnet " \
              + f"\"{feats}\" ark:- | " \
              + f"latgen-faster  --max-active={max_active} --max-mem={max_mem} --beam={beam} --lattice-beam={lattice_beam} " \
              + f"--acoustic-scale={acwt} --allow-partial=true --word-symbol-table={graphdir}/words.txt " \
              + f"{graphdir}/TLG.fst ark:- \"ark:|gzip -c > {_dir}/lat.1.gz\" ")
