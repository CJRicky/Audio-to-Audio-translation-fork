import re

def split_str_from_words(l, s):
    m = re.split(rf"({'|'.join(l)})", s)
    print(m)
    return [i for i in m if i] # removes empty strings (improvements are welcome)

my_strings = split_str_from_words(["speaker1", "speaker2"], "speaker1 helloworldhello \
speaker2 hejsvejs. Vad heter du? speaker1. Inte mycket.")

print(my_strings)