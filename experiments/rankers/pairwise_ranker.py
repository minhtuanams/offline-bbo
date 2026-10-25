import re
from experiments.rankers.ranker import Ranker
from experiments.utils.utils import str_to_discrete, discrete_to_str

TASK_DESCRIPTION = '''
    You aim to find a new length-8 DNA sequence, composed of A, C, G, and T,
    with maximum binding affinity with the transcription factor SIX6_REF_R1.
    You are given some ... examples and oracle score of 
'''

PAIRWISE_INSTRUCTION = '''
    Given the two candidate sequences, decide which one has the higher predicted binding affinity.
    Output only that sequence, no explanation.
'''


class PairwiseRanker(Ranker):
    """Merge sort where each comparison is one LLM call on a pair of candidates."""

    def __init__(self, lm, sampler, description=TASK_DESCRIPTION, instruction=PAIRWISE_INSTRUCTION):
        self.lm = lm
        self.sampler = sampler
        self.description = description
        self.instruction = instruction

    def format_prompt(self, samples, seq_a, seq_b):
        in_context = "\n".join(
            f"DNA Sequence: {seq}, Binding Affinity: {score:.2f}"
            for seq, score in samples
        )
        return (
            f"Task Description:\n{self.description}\n\n"
            f"Offline Dataset:\n{in_context}\n\n"
            f"Instruction:\n{self.instruction}\n\n"
            f"DNA Sequence: {seq_a}\n"
            f"DNA Sequence: {seq_b}"
        )

    def pair_rank(self, seq_a, seq_b):
        '''Return [better, worse]; keeps the input order if the reply is unparsable.'''
        # Examples close to this specific pair
        X_s, y_s = self.sampler.sample(ref=str_to_discrete([seq_a, seq_b]))
        samples = list(zip(discrete_to_str(X_s), y_s.tolist()))

        response = self.lm(self.format_prompt(samples, seq_a, seq_b))
        for s in re.findall(r"[ACGT]{8}", response.upper()):
            if s == seq_a:
                return [seq_a, seq_b]
            if s == seq_b:
                return [seq_b, seq_a]
        return [seq_a, seq_b]

    def rank(self, cands_str):
        return self.merge_rank(list(cands_str))

    def merge_rank(self, items):
        if len(items) <= 1:
            return items
        mid = len(items) // 2
        left, right = self.merge_rank(items[:mid]), self.merge_rank(items[mid:])

        merged, i, j = [], 0, 0
        while i < len(left) and j < len(right):
            if self.pair_rank(left[i], right[j])[0] == left[i]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
        return merged + left[i:] + right[j:]
