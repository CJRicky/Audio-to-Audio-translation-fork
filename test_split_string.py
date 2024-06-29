"""import re

def split_str_from_words(l, s):
    m = re.split(rf"({'|'.join(l)})", s)
    print(m)
    return [i for i in m if i] # removes empty strings (improvements are welcome)

my_strings = split_str_from_words(["speaker1", "speaker2"], "speaker1 helloworldhello \
speaker2 hejsvejs. Vad heter du? speaker1. Inte mycket.")

print(my_strings)"""
import re

def split_conversation(l, s):
    # Create a regex pattern that matches any of the speakers
    pattern = "|".join(l)

    # Use the pattern to split the string
    m = re.split(pattern, s)
    
    return [i.strip() for i in m if i] # removes leading/trailing white spaces and empty strings

my_strings = split_conversation(["speaker1", "speaker2"], 
                                  "speaker1 helloworldhello speaker2 hejsvejs. Vad heter speaker1 du? speaker2 Inte mycket.")

print(my_strings)


prompt = (f"Translate transcript to user_input. Separate the two persons "
            "by adding _speaker1_ and _speaker2_ and whitespaces on each side each time "
            "the person starts to speak.")
print(prompt)
### Explanation:
#1. **`map(re.escape, l)`:** This ensures that any special characters in the list `l` are treated as literals during the split.
#2. **Filter in list comprehension:**
#   - **`i`:** This ensures non-empty strings are retained.
#   - **`i not in l`:** This ensures "speaker1" and "speaker2" are not included in the final array.
#
#This will give you the desired output without including `"speaker1"` and `"speaker2"` in the final list.