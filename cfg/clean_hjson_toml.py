import os
from glob import glob

for file in glob("./*/*.toml"):
    os.remove(file)

for file in glob("./*/*.hjson"):
    os.remove(file)
