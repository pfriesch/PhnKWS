import os
import random

import kaldi.fstext as fst

from utils.logger_config import logger
from utils.utils import run_shell

# #KWS_001
# 0	1	72070	72070
# 1	2	117093	117093
# 2
#
# KWS_002
# 0	1	156223	156223
# 1	2	198047	198047
# 2
#
# KWS_003
# 0	1	33224	33224
# 1
#
# KWS_004
# 0	1	185382	185382
# 1
#
# KWS_005
# 0	1	194651	194651
# 1	2	62931	62931
# 2
#
# KWS_006
# 0	1	49768	49768
# 1	2	40836	40836
# 2


isymbols = fst.SymbolTable()

words = ["MARVIN", "ALEXA", "FORWARD", "MOVE"]
keywords = ["MARVIN", "ALEXA", "FORWARD", ["MOVE", "FORWARD"]]

isymbols.set_name("words")
isymbols.add_symbol("<eps>")
isymbols.add_symbol("<UNK>")
for word in words:
    isymbols.add_symbol(word)
    print(isymbols.find_index(word))

isymbols.write_text("words.syms")
# print(isymbols.num_symbols())
# isymbols.
weight = fst.TropicalWeight(-1)

fsts = {}

word_fst = fst.StdVectorFst()
word_fst.set_input_symbols(isymbols)
word_fst.set_output_symbols(isymbols)
start_state = word_fst.add_state()
word_fst.set_start(start_state)
sym = isymbols.find_index("<UNK>")

word_out_state = word_fst.add_state()
word_fst.add_arc(start_state, fst.StdArc(sym, sym, weight, word_out_state))
word_fst.set_final(word_out_state)

fsts["<UNK>"] = word_fst

for kw in keywords:
    if isinstance(kw, str):
        word = kw
        print(word)

        word_fst = fst.StdVectorFst()
        word_fst.set_input_symbols(isymbols)
        word_fst.set_output_symbols(isymbols)
        start_state = word_fst.add_state()
        word_fst.set_start(start_state)
        sym = isymbols.find_index(word)
        print("in: ", sym)

        word_out_state = word_fst.add_state()
        word_fst.add_arc(start_state, fst.StdArc(sym, sym, weight, word_out_state))
        word_fst.set_final(word_out_state)

        fsts[word] = word_fst

        print(word_fst)
        word_fst.draw(f"fst_test_{word}.dot", portrait=True)
        run_shell(f"dot -Tpdf fst_test_{word}.dot -o fst_test_{word}.pdf")

        # word_out_state = word_fst.add_state()
        # word_fst.add_arc(start_state, fst.StdArc(random.randint(0, 3), 0, fst.TropicalWeight(-1), word_out_state))
        # word_fst.set_final(word_out_state)
    elif isinstance(kw, list):
        word_fst = fst.StdVectorFst()
        word_fst.set_input_symbols(isymbols)
        word_fst.set_output_symbols(isymbols)
        prev_state = word_fst.add_state()
        word_fst.set_start(prev_state)
        for word in kw:
            sym = isymbols.find_index(word)
            word_out_state = word_fst.add_state()
            word_fst.add_arc(prev_state, fst.StdArc(sym, sym, weight, word_out_state))
            prev_state = word_out_state

        word_fst.set_final(prev_state)
        word = "_".join(kw)
        fsts[word] = word_fst

        print(word_fst)
        word_fst.draw(f"fst_test_{word}.dot", portrait=True)
        run_shell(f"dot -Tpdf fst_test_{word}.dot -o fst_test_{word}.pdf")

    else:
        raise RuntimeError

union_fst_name = "union"
union_fst = None
for word, wfst in fsts.items():
    if union_fst is None:
        union_fst = wfst
    else:
        union_fst.union(wfst)

print(union_fst)
union_fst = union_fst.rmepsilon()
print(union_fst)

union_fst = fst.determinize(union_fst)
print(union_fst)

union_fst.draw(f"fst_test_{union_fst_name}.dot", portrait=True)
run_shell(f"dot -Tpdf fst_test_{union_fst_name}.dot -o fst_test_{union_fst_name}.pdf")

# print(word_fst)
# print("")
