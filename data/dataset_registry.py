from data.kaldi_dataset_framewise import KaldiDatasetFramewise
from data.kaldi_dataset_unaligned import KaldiDatasetUnaligned


def get_dataset(feature_dict, label_dict, phn_mapping, context_left, context_right, max_sequence_length,
                framewise_labels,
                tensorboard_logger):
    if framewise_labels:
        dataset = KaldiDatasetFramewise(feature_dict, label_dict, context_left, context_right, max_sequence_length,
                                        tensorboard_logger)
    else:
        dataset = KaldiDatasetUnaligned(feature_dict, label_dict, phn_mapping, context_left, context_right,
                                        max_sequence_length,
                                        tensorboard_logger)
    return dataset
