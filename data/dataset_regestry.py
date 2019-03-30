from data.datasets import DatasetType
from data.datasets.kaldi_dataset_framewise_sequential import KaldiDatasetFramewise
from data.datasets.kaldi_dataset_framewise_sequential_context import KaldiDatasetFramewiseContext
from data.datasets.kaldi_dataset_framewise_shuffled_frames import KaldiDatasetFramewiseContextShuffledFrames
from data.datasets.kaldi_dataset_sequential import KaldiDatasetSequential
from data.datasets.kaldi_dataset_sequential_context import KaldiDatasetSequentialContext
from utils.logger_config import logger


def get_dataset(dataset_type,
                data_cache_root,
                dataset_name,
                feature_dict,
                label_dict,
                max_sample_len=None,
                left_context=0,
                right_context=0,
                normalize_features=True,
                phoneme_dict=None,

                max_seq_len=100,
                max_label_length=None,
                overfit_small_batch=False
                ):
    assert DatasetType[dataset_type]
    dataset_type = DatasetType[dataset_type]
    if dataset_type == DatasetType.FRAMEWISE_SHUFFLED_FRAMES:
        assert max_sample_len == None
        assert max_seq_len == None
        assert max_label_length == None

        assert 'lab_cd' in label_dict and 'lab_mono' in label_dict

        return KaldiDatasetFramewiseContextShuffledFrames(data_cache_root, dataset_name, feature_dict, label_dict,
                                                          left_context, right_context, normalize_features,
                                                          overfit_small_batch)

    elif dataset_type == DatasetType.FRAMEWISE_SEQUENTIAL:
        # assert phoneme_dict == None
        assert 'lab_cd' in label_dict or 'lab_mono' in label_dict or 'lab_phnframe' in label_dict

        return KaldiDatasetFramewise(data_cache_root, dataset_name, feature_dict, label_dict, phoneme_dict,
                                     max_sample_len,
                                     left_context, right_context, normalize_features, max_seq_len, max_label_length,
                                     overfit_small_batch)

    elif dataset_type == DatasetType.FRAMEWISE_SEQUENTIAL_APPENDED_CONTEXT:
        # assert phoneme_dict == None
        assert 'lab_cd' in label_dict or 'lab_mono' in label_dict

        # assert left_context > 0 or right_context > 0
        label_index_from = 0  # TODO temporary for lstm
        logger.warn("setting label_index_from to 0 (temporarly for lstm)")

        return KaldiDatasetFramewiseContext(data_cache_root, dataset_name, feature_dict, label_dict, max_sample_len,
                                            left_context, right_context, normalize_features, max_seq_len,
                                            max_label_length,
                                            overfit_small_batch, label_index_from)
    elif dataset_type == DatasetType.SEQUENTIAL:
        assert 'lab_phn' in label_dict

        return KaldiDatasetSequential(data_cache_root, dataset_name, feature_dict, label_dict, phoneme_dict,
                                      max_sample_len, left_context, right_context, normalize_features, max_seq_len,
                                      max_label_length, overfit_small_batch)
    elif dataset_type == DatasetType.SEQUENTIAL_APPENDED_CONTEXT:
        assert 'lab_phn' in label_dict

        if not (left_context > 0 or right_context > 0):
            logger.warn("No context but context given")

        return KaldiDatasetSequentialContext(data_cache_root, dataset_name, feature_dict, label_dict, phoneme_dict,
                                             max_sample_len, left_context, right_context, normalize_features,
                                             max_seq_len, max_label_length, overfit_small_batch)
    else:
        raise ValueError
