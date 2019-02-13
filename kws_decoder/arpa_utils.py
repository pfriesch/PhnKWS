import math

UNK_WORD = "<UNK>"


def make_kw_arpa(keywords):
    ""

    unk_word = "<UNK>"
    start_symbol = "<s>"
    stop_symbol = "</s>"

    ngrams = {'unigrams': [],
              'bigrams': []}

    ngrams['unigrams'].append(start_symbol)
    ngrams['unigrams'].append(stop_symbol)
    ngrams['unigrams'].append(unk_word)

    for kw in keywords:
        if " " in kw:
            # TODO This case is not porperly handled yet with <s> and </s> and unigram etc. I have to think about it
            raise NotImplementedError("This case is not porperly handled yet with <s> and </s>")
            kw_split = kw.split(" ")
            assert len(kw_split) == 2
            ngrams['bigrams'].append(kw)

            ngrams['unigrams'].append(kw_split[0])
            ngrams['unigrams'].append(kw_split[1])

        else:
            ngrams['unigrams'].append(kw)

    num_unigram = len(ngrams['unigrams'])
    unigram_prob = math.log10(1.0 / num_unigram)  # probably does not need to sum to 1

    # -4 for <s></s>
    num_bigram = len(ngrams['bigrams']) + (2 * len(ngrams['unigrams']) - 4)
    bigram_prob = math.log10(1.0 / num_bigram)  # probably does not need to sum to 1

    out_lines = []
    out_lines.append(f"")
    out_lines.append("\\data\\")
    out_lines.append(f"ngram 1={num_unigram}")
    out_lines.append(f"ngram 2={num_bigram}")
    out_lines.append(f"")
    out_lines.append(f"\\1-grams:")
    for kw in ngrams['unigrams']:
        out_lines.append(f"{unigram_prob}\t{kw}")

    out_lines.append(f"")
    out_lines.append(f"\\2-grams:")

    if len(ngrams['bigrams']) > 0:
        # TODO  no backoff prob since we do not want the case of half uttered keywords
        # (is this statement true and we are sure we do not want that?)
        for kw in ngrams['bigrams']:
            out_lines.append(f"{bigram_prob}\t{kw}")

    for kw in ngrams['unigrams']:
        if kw not in [start_symbol, stop_symbol]:
            out_lines.append(f"{bigram_prob}\t{start_symbol} {kw}")
            out_lines.append(f"{bigram_prob}\t{kw} {stop_symbol}")

    out_lines.append(f"")
    out_lines.append("\\end\\")
    out_lines.append(f"")

    return "\n".join(out_lines)
