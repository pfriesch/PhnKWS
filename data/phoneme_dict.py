import collections

PhonemeDict = collections.namedtuple("PhonemeDict", ["idx2phoneme", "idx2reducedIdx", "phoneme2reducedIdx"])


def load_phoneme_dict(idx2phoneme, idx2reducedIdx, phoneme2reducedIdx):
    idx2phoneme = {int(k): v for k, v in idx2phoneme.items()}
    idx2reducedIdx = {int(k): v for k, v in idx2reducedIdx.items()}
    return PhonemeDict(idx2phoneme, idx2reducedIdx, phoneme2reducedIdx)


def get_phoneme_dict(phoneme_path, stress_marks=False, word_position_dependency=False):
    with open(phoneme_path, "r") as f:
        phoneme_list = f.readlines()

    # remove all disambiguation symbols
    phoneme_list = [line.strip().split(" ") for line in phoneme_list if not "#" in line]

    idx2phoneme = {int(idx): phoneme for phoneme, idx in phoneme_list}

    # can't have word_position_dependency without stress_marks
    assert (stress_marks and word_position_dependency) or stress_marks or (
            not stress_marks and not word_position_dependency)

    if not word_position_dependency:
        idx2phoneme = {idx: phoneme.split("_")[0] for idx, phoneme in idx2phoneme.items()}

    if not stress_marks:
        idx2phoneme = {idx: ''.join([i for i in phoneme if not i.isdigit()]) for idx, phoneme in idx2phoneme.items()}

    # avoiding set here to perserve order
    phoneme2reducedIdx = {phoneme: idx for idx, phoneme in
                          enumerate(list(collections.OrderedDict.fromkeys(idx2phoneme.values()).keys()))}

    idx2reducedIdx = {idx: phoneme2reducedIdx[phoneme] for idx, phoneme in idx2phoneme.items()}

    return PhonemeDict(idx2phoneme, idx2reducedIdx, phoneme2reducedIdx)

# if __name__ == '__main__':
# get_dict("/mnt/data/libs/kaldi/egs/librispeech/s5/data/lang/phones.txt")
# get_dict("/mnt/data/libs/kaldi/egs/timit/s5/data/lang/phones.txt")
