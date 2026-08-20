from handwriting_rnn import RNNHandwritingGenerator
from handwriting_ebb import EBB_DEFAULTS

# IMPORTANT: paste your actual TEXT from preview_document.py here, exactly
# as it is there, so this checks the real content that crashed.
TEXT = """
The difference between a DFA and an NFA is that a DFA seems to be a finite acceptor in which every transition state must be defined for each internal state. For example, let's say that we have an alphabet that accepts just a set of a and b. Each internal state needs to be defined as what it would do if it got an a or b input. As for an NFA, you don’t need to explain every transition state for the internal states. You can just define the state that you need. Also, NFAs allow for things such as lamba transitions, enabling secondary arguments. When it comes to what I find more straightforward to construct for me, DFA’s reason is that, for NFA’s, I just have a hard time knowing when to use a lambda function to skip processes. And for DFA, all I know is that I have to define what to do for each input for an internal state. When converting an NFA to a DFA, I use the table method since it helps me keep track of everything as I go.  


Procedure Bob is used mainly in this course to convert an NFA to a DFA that allows only one accepting state. Procedure Mark reduces the number of states that a DFA contains.
"""

cfg = dict(EBB_DEFAULTS)
gen = RNNHandwritingGenerator(cfg, rng_seed=1, bias=0.75, style=9)

lines = gen.wrap_text(TEXT)
print(f"wrap_text produced {len(lines)} lines:\n")
for i, line in enumerate(lines):
    sanitized = gen._sanitize_for_rnn(line)
    flag = "  <-- OVER 110!" if len(line) > 110 else ""
    sanitize_flag = "  <-- sanitize CHANGED length!" if len(sanitized) != len(line) else ""
    print(f"  [{i}] len={len(line)}{flag} sanitized_len={len(sanitized)}{sanitize_flag}")
    print(f"      {line!r}")
