import re
from experiments.rankers.ranker import Ranker
from experiments.utils.utils import discrete_to_str

TASK_DESCRIPTION = '''
    You aim to find a new length-8 DNA sequence, composed of A, C, G, and T,
    with maximum binding affinity with the transcription factor SIX6_REF_R1.
'''

INSTRUCTION = '''
    Given the new candidate sequences, rank them in order of predicted binding affinity from highest to lowest.
    Output only the rankings the new sequences, one sequence per line, no numbering, no explanation.
'''

class FullRanker(Ranker):
    """Rank all candidates in a single LLM call."""

    def __init__(self, lm, sampler, description=TASK_DESCRIPTION, instruction=INSTRUCTION):
        self.lm = lm
        self.sampler = sampler
        self.description = description
        self.instruction = instruction

    def format_prompt(self, samples, cands_str):
        '''Task description + in-context samples + instruction + candidates.'''
        in_context = "\n".join(
            f"DNA Sequence: {seq}, Binding Affinity: {score:.2f}"
            for seq, score in samples
        )
        candidates = "\n".join(f"DNA Sequence: {seq}" for seq in cands_str)

        return (
            f"Task Description:\n{self.description}\n\n"
            f"Offline Dataset:\n{in_context}\n\n"
            f"Instruction:\n{self.instruction}\n\n"
            f"{candidates}"
        )

    def parse_ranking(self, response, cands_str):
        '''Read ranked sequences; drop invalid/duplicates, append any missing.'''
        valid, seen, ranking = set(cands_str), set(), []
        for s in re.findall(r"[ACGT]{8}", response.upper()):
            if s in valid and s not in seen:
                ranking.append(s)
                seen.add(s)
        return ranking + [s for s in cands_str if s not in seen]

    def rank(self, cands_str):
        X_s, y_s = self.sampler.sample()
        samples = list(zip(discrete_to_str(X_s), y_s.tolist()))

        response = self.lm(self.format_prompt(samples, cands_str))
        return self.parse_ranking(response, cands_str)
