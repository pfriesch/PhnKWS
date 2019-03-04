from utils.logger_config import logger


def int2sym(transcript_file, mapping_file):
    with open(mapping_file, "r") as f:
        mapping = f.readlines()
    mapping = dict([m.strip().split(" ") for m in mapping])

    inv_mapping = {_id: phn for phn, _id in mapping.items()}

    with open(transcript_file, "r") as f:
        transcripts = f.readlines()

    transcripts = [t.strip().split(" ", 1) for t in transcripts]
    mapped_transcripts = {}
    # _mapped_transcripts = []

    # for t in transcripts:
    #     if len(t) != 2:
    #         mapped_transcripts[t] = ["---"]
    #         transcripts.remove(t)

    for _idx, tran in enumerate(transcripts):
        if len(tran) == 1:
            transcripts[_idx] = (tran, str(mapping['<UNK>']))
            logger.debug(f"transcript of {tran} was empty, using <UNK> instead")

    for sample_id, transcript in transcripts:
        if sample_id.endswith("-1"):
            sample_id = sample_id[:-2]
        assert sample_id not in mapped_transcripts
        mapped_transcripts[sample_id] = [inv_mapping[transcript]]
        # _mapped_transcripts.append((sample_id, inv_mapping[transcript]))

    return mapped_transcripts
