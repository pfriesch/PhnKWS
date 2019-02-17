import os
import random

import kaldi.fstext as fst

from utils.logger_config import logger
from utils.utils import run_shell

isymbols = fst.SymbolTable()

words = {1: "MARVIN", 2: "ALEXA", 3: "FORWARD"}

isymbols.set_name("words")
isymbols.add_symbol("<eps>")
isymbols.add_symbol("<UNK>")
for word in words:
    isymbols.add_symbol(word)

isymbols.write_text("words.syms")

gfst = fst.StdVectorFst()
gfst.set_input_symbols(isymbols)
gfst.set_output_symbols(isymbols)
start_state = gfst.add_state()
gfst.set_start(start_state)

for word in words:
    word_out_state = gfst.add_state()
    gfst.add_arc(start_state, fst.StdArc(random.randint(0, 3), 0, fst.TropicalWeight(-1), word_out_state))
    gfst.set_final(word_out_state)

    word_out_state = gfst.add_state()
    gfst.add_arc(start_state, fst.StdArc(random.randint(0, 3), 0, fst.TropicalWeight(-1), word_out_state))
    gfst.set_final(word_out_state)

gfst.draw("fst_test.dot", portrait=True)
run_shell("dot -Tpdf fst_test.dot -o fst_test.pdf")
print(gfst)
print("")
