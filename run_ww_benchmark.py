#
# Copyright 2018 Picovoice Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import logging
import multiprocessing
import os
import time

# # Filter out logs from sox.
# logging.getLogger('sox').setLevel(logging.ERROR)
# logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
# 
# parser = argparse.ArgumentParser(description='Benchmark for different wake-word engines')
# 
# parser.add_argument(
#     '--common_voice_directory',
#     type=str,
#     help='root directory of Common Voice dataset',
#     required=True)
# 
# parser.add_argument(
#     '--alexa_directory',
#     type=str,
#     help='root directory of Alexa dataset',
#     required=True)
# 
# parser.add_argument(
#     '--demand_directory',
#     type=str,
#     help='root directory of Demand dataset',
#     required=True)
# 
# parser.add_argument(
#     '--output_directory',
#     type=str,
#     help='output directory to save the results')
# 
# parser.add_argument(
#     '--add_noise',
#     action='store_true',
#     default=False,
#     help='add noise to the datasets')

from ww_benchmark.wakeword_executor import WakeWordExecutor, Dataset, Datasets, CompositeDataset, csv


def run_detection(engine):
    """
    Run wake-word detection for a given engine.

    :param engine_type: type of the engine.
    :return: tuple of engine and list of accuracy information for different detection sensitivities.
    """

    res = []
    for sensitivity in engine.sensitivity_range:
        start_time = time.process_time()

        executor = WakeWordExecutor(engine, sensitivity, keyword, dataset, noise_dataset=noise_dataset)
        false_alarm_per_hour, miss_rate = executor.execute()
        executor.release()

        end_time = time.process_time()

        logging.info('[%s][%s] took %s minutes to finish', engine.name, sensitivity, (end_time - start_time) / 60)

        res.append(dict(sensitivity=sensitivity, false_alarm_per_hour=false_alarm_per_hour, miss_rate=miss_rate))

    return engine.value, res


def run(
        model="/mnt/data/pytorch-kaldi/exp_TIMIT_MLP_FBANK/TIMIT_MLP_fbank_20190202_170357r-_20190202_174619r-_20190202_183621",
        keyword='alexa',
        common_voice_directory="/mnt/data/pytorch-kaldi/bench_data/cv_corpus_v1",
        alexa_directory="/mnt/data/pytorch-kaldi/bench_data/alexa",
        demand_directory="/mnt/data/pytorch-kaldi/bench_data/demand",
        add_noise=False):
    """
    Benchmark for different wake-word engines

    :param model:
    :param keyword:
    :param common_voice_directory: root directory of Common Voice dataset
    :param alexa_directory: root directory of Alexa dataset
    :param demand_directory: root directory of Demand dataset
    :param add_noise: add noise to the datasets
    :return:
    """
    engines = []  # TODO

    background_dataset = Dataset.create(Datasets.COMMON_VOICE, common_voice_directory, exclude_words=keyword)
    logging.info('loaded background speech dataset with %d examples' % background_dataset.size())

    keyword_dataset = Dataset.create(Datasets.ALEXA, alexa_directory)
    logging.info('loaded keyword dataset with %d examples' % keyword_dataset.size())

    if add_noise:
        noise_dataset = Dataset.create(Datasets.DEMAND, demand_directory)
        logging.info('loaded noise dataset with %d examples' % noise_dataset.size())
    else:
        noise_dataset = None

    # Interleave the keyword dataset with background dataset to simulate the real-world conditions.
    dataset = CompositeDataset(datasets=(background_dataset, keyword_dataset), shuffle=True)

    # Run the benchmark for each engine in it's own process.
    with multiprocessing.Pool() as pool:
        results = pool.map(run_detection, engines)

    # Save the results.
    if output_directory:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for engine, result in results:
            with open(os.path.join(output_directory, '%s.csv' % engine), 'w') as f:
                writer = csv.DictWriter(f, ['sensitivity', 'false_alarm_per_hour', 'miss_rate'])
                writer.writeheader()
                writer.writerows(result)


if __name__ == '__main__':
    run()
