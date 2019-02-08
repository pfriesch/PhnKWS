# install espeak:  sudo apt-get install festival espeak
import collections
import json
import os

from phonemizer.phonemize import phonemize
from tqdm import tqdm

Separator = collections.namedtuple('Separator', ['word', 'syllable', 'phone'])


def text2phones(text_path):
    with open(text_path, "r") as file:
        text_lines = file.readlines()

    total_phn_count = collections.Counter()
    phn_transcripts = []

    for line in tqdm(text_lines):
        id, txt = line.strip().split(" ", 1)
        all_phn = phonemize(txt, language='en-us', backend='espeak',
                            # separator=Separator(' ', '===', '_p_'), strip=False)
                            separator=Separator(' ', '-', '_'), strip=False)
        words = all_phn.split(" ")
        total_phn_count[' '] += words.count(' ')
        for word in words:
            phns = word.split("_")
            for phn in phns:
                total_phn_count[phn] += 1

        words = all_phn.split(" ")
        phn_transcript = []
        for word in words:
            phn_transcript.append(" ")
            if word != '':
                phns = word.split("_")
                for phn in phns:
                    if phn != '':
                        phn_transcript.append(phn)
        phn_transcripts.append(
            {"id": id, "txt": txt.lower(), "phn": phn_transcript})

    all_phns = sorted(list(total_phn_count.keys()))
    all_phns.remove("")
    id, all_phns = zip(*enumerate(all_phns))
    # phn_mapping_id = dict(zip(id, all_phns))
    # phn_mapping = dict(zip(all_phns, id))

    # return phn_transcripts, phn_mapping, phn_mapping_id, total_phn_count
    return phn_transcripts, total_phn_count


def save_phones(name, txt_file, save_mapping=True):
    # phn_transcripts, phn_mapping, phn_mapping_id, total_phn_count = text2phones(txt_file)
    phn_transcripts, total_phn_count = text2phones(txt_file)
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)
    json.dump(phn_transcripts, open("phn_transcripts.json", "w"))
    json.dump(total_phn_count, open("total_phn_count.json", "w"))
    # if save_mapping:
    #     json.dump(phn_mapping, open("phn_mapping.json", "w"))
    #     json.dump(phn_mapping_id, open("phn_mapping_id.json", "w"))


if __name__ == '__main__':
    save_phones(name="test_clean",
                txt_file="/mnt/data/libs/kaldi/egs/librispeech/s5/data/test_clean/text",
                save_mapping=False)
