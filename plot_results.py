import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from ww_benchmark.wakeword_executor import plot_single_alexa_statistics, get_FAH_MISS, plot_single_multi_kw_statistics, \
    plot_single_alexa_statistics_ROC, KWResult


def plot_roc(results_path, unk_sensitivity=0.0):
    with open(results_path, "r") as f:
        print(f"loading {results_path}")
        results = json.load(f)
        results['results'] = list({r[0] + str(r[2]): r for r in results['results']}.values())

    # if "ALEXA" in results['keywords']:
    #     plot_single_alexa_statistics_ROC(results, results_save_folder)
    #     # plot_single_alexa_statistics_filter_posterior(results, self.results_save_folder)
    # else:
    #     raise NotImplementedError
    #     plot_single_multi_kw_statistics(results)

    # NORMALIZE KW
    min = float('inf')
    max = -float('inf')
    for result in results['results']:
        result = KWResult(*result)

        if result.detected_kw[0] == 'ALEXA':
            if result.acoustic_posterior < min:
                min = result.acoustic_posterior
            if result.acoustic_posterior > max:
                max = result.acoustic_posterior

    max_new = max - min
    for result in results['results']:
        mapped_result = KWResult(*result)

        if mapped_result.detected_kw[0] == 'ALEXA':
            result[7] = (result[7] - min) / max_new
            assert 0 <= result[7] <= 1

    # NORMALIZE UNK
    min = float('inf')
    max = -float('inf')
    for result in results['results']:
        result = KWResult(*result)

        if result.detected_kw[0] == '<UNK>':
            if result.acoustic_posterior < min:
                min = result.acoustic_posterior
            if result.acoustic_posterior > max:
                max = result.acoustic_posterior

    max_new = max - min
    for result in results['results']:
        mapped_result = KWResult(*result)

        if mapped_result.detected_kw[0] == '<UNK>':
            result[7] = (result[7] - min) / max_new
            assert 0 <= result[7] <= 1

    # plot_single_alexa_statistics_ROC(results,
    #                                  "/Volumes/SD/projects2/KTH/IndividualCourse/phn_keywordspotting/pytorch-kaldi/tmp")

    graph_ww, graph_unk = get_FAH_MISS(results, filter_posterior=True, unk_sensitivity=unk_sensitivity)
    # graph_ww = np.array(graph_ww)
    # graph_unk = np.array(graph_unk)
    return graph_ww, graph_unk
    # # false_alarm_per_hour:0, miss_rate:1
    # plt.plot(graph_ww[:, 0], graph_ww[:, 1])
    # plt.xlabel("false alarms per hour")
    # plt.ylabel("Miss rate")
    # plt.plot([10.6], [0.0514], "+")
    #
    # plt.show()
    #
    # plt.plot(graph_unk[:, 0], graph_unk[:, 1])
    # plt.xlabel("false alarms per hour")
    # plt.ylabel("Miss rate")
    # plt.plot([10.6], [0.0514], "+")
    # axes = plt.axes()
    # axes.set_xlim([0, 20])
    # plt.show()
    # print("")


def main():
    results_path_lstm = "/Volumes/SD/projects2/KTH/IndividualCourse/phn_keywordspotting/pytorch-kaldi/trained_models/libri_LSTM_fbank_ce/alexa_results/results_snr_db_994.json"
    graph_ww_lstm, graph_unk_lstm = plot_roc(results_path_lstm, unk_sensitivity=0.0)
    results_path_WN_ce = "/Volumes/SD/projects2/KTH/IndividualCourse/phn_keywordspotting/pytorch-kaldi/trained_models/libri_WaveNetBig_ce/libri_WaveNetBIG_fbank_ce/alexa_results/results_snr_db_994.json"
    graph_ww_WN_ce, graph_unk_WN_ce = plot_roc(results_path_WN_ce, unk_sensitivity=0.03)
    results_path_WN_ctc = "/Volumes/SD/projects2/KTH/IndividualCourse/phn_keywordspotting/pytorch-kaldi/trained_models/libri_WaveNetBig_ctc/libri_WaveNetBIG_fbank_ctc_PER_21percent/alexa_results/results_snr_db_994.json"
    graph_ww_WN_ctc, graph_unk_WN_ctc = plot_roc(results_path_WN_ctc, unk_sensitivity=0.035)

    # results_save_folder = "/Volumes/SD/projects2/KTH/IndividualCourse/phn_keywordspotting/pytorch-kaldi/tmp"
    plt.clf()
    for g in graph_ww_lstm:
        data = np.array(graph_ww_lstm[g])
        data = np.array(list(filter(lambda x: x[0] < 10, data)))
        print("LSTM", data[data[:, 0].searchsorted([2])], data[data[:, 0].searchsorted([2]) - 1])

        plt.plot(data[:, 0], data[:, 1],
                 # label=f"lstm _{g}(AUC: {auc(data[:, 0], data[:, 1]):.02f})")
                 label=f"LSTM (AUC: {auc(data[:, 0], data[:, 1]):.02f})")
    #
    # # false_alarm_per_hour:0, miss_rate:1
    # plt.plot(graph_ww_lstm[:, 0], graph_ww_lstm[:, 1],
    #          # label=f"lstm (AUC: {auc(graph_ww_lstm[:, 0], graph_ww_lstm[:, 1]):.03f})")
    #          label=f"lstm")
    # # plt.plot([10.6], [0.0514], "+", label="lstm_p")
    #
    # plt.plot(graph_ww_WN_ce[:, 0], graph_ww_WN_ce[:, 1],
    #          # label=f"WN_ce (AUC: {auc(graph_ww_WN_ce[:, 0], graph_ww_WN_ce[:, 1]):.03f})")
    #          label=f"WN_ce")
    # # plt.plot([5.31], [0.0948], "x", label="WN_ce_p")
    #
    # plt.plot(graph_ww_WN_ctc[:, 0], graph_ww_WN_ctc[:, 1],
    #          # label=f"WN_ctc (AUC: {auc(graph_ww_WN_ctc[:, 0], graph_ww_WN_ctc[:, 1]):.03f})")
    #          label=f"WN_ctc")
    # # plt.plot([3.54], [0.1246], "o", label="WN_ctc_p")
    # plt.plot([10.6], [0.0514], "+")
    #
    # plt.xlabel("false alarms per hour")
    # plt.ylabel("Miss rate")
    # plt.legend()
    # axes = plt.axes()
    # axes.set_xlim([-0.1, 15])
    # axes.set_ylim([-0.0001, 0.3])
    #
    # plt.show()

    for g in graph_ww_WN_ce:
        data = np.array(graph_ww_WN_ce[g])
        data = np.array(list(filter(lambda x: x[0] < 10, data)))
        print("WN_ce_", data[data[:, 0].searchsorted([2])], data[data[:, 0].searchsorted([2]) - 1])

        plt.plot(data[:, 0], data[:, 1],
                 # label=f"WN_ce_ _{g}(AUC: {auc(data[:, 0], data[:, 1]):.03f})")
                 label=f"Residual-Dilated-Gated-CNN CE (AUC: {auc(data[:, 0], data[:, 1]):.02f})")

    # plt.plot([5.31], [0.0948], "+")
    #
    # plt.xlabel("false alarms per hour")
    # plt.ylabel("Miss rate")
    # plt.legend()
    # axes = plt.axes()
    # axes.set_xlim([-0.1, 15])
    # axes.set_ylim([-0.0001, 0.3])
    #
    # plt.show()

    for g in graph_ww_WN_ctc:
        data = np.array(graph_ww_WN_ctc[g])
        data = np.array(list(filter(lambda x: x[0] < 10, data)))
        print("WN_ctc__", data[data[:, 0].searchsorted([2])], data[data[:, 0].searchsorted([2]) - 1])

        plt.plot(data[:, 0], data[:, 1],
                 # label=f"WN_ctc__{g} (AUC: {auc(data[:, 0], data[:, 1]):.03f})")
                 label=f"Residual-Dilated-Gated-CNN CTC (AUC: {auc(data[:, 0], data[:, 1]):.02f})")

    # plt.plot([3.54], [0.1246], "+")
    plt.xlabel("False Alarms per Hour")
    plt.ylabel("Miss Rate")
    plt.legend()
    axes = plt.axes()
    axes.set_xlim([-0.1, 10])
    axes.set_ylim([-0.0001, 0.3])

    # plt.show()
    plt.savefig("ROC.png")
    plt.savefig("ROC.pdf")

    # plt.plot(graph_unk_lstm[:, 0], graph_unk_lstm[:, 1], label="lstm")
    # plt.plot(graph_unk_WN_ce[:, 0], graph_unk_WN_ce[:, 1], label="WN_ce")
    # plt.plot(graph_unk_WN_ctc[:, 0], graph_unk_WN_ctc[:, 1], label="WN_ctc")
    # plt.xlabel("False Alarms per Hour")
    # plt.ylabel("Miss Rate")
    # plt.legend()
    #
    # # plt.plot([10.6], [0.0514], "+")
    # axes = plt.axes()
    # axes.set_xlim([-0.1, 20])
    # axes.set_ylim([-0.0001, 0.3])
    # plt.show()
    # print("")

    # with open(results_path[:-5] + "_filter_posterior.json", "w") as f:
    #     print(get_FAH_MISS(results))


if __name__ == '__main__':
    main()
