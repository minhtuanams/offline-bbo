from experiments.rankers.ranker import Ranker
from experiments.utils.utils import str_to_discrete, discrete_to_str
import torch


TASK_DESCRIPTION = '''
    You aim to find a new length-8 DNA sequence, composed of A, C, G, and T,
    with maximum binding affinity with the transcription factor SIX6_REF_R1.
    You are given some ... examples and oracle score of 
'''


PAIRWISE_PROB_INSTRUCTION = '''
    Given the two candidate sequences, decide which one has the higher predicted binding affinity.
    Does the first DNA sequence have higher predicted binding affinity than the second?
    Output strictly YES or NO. Output YES if the first sequence has higher binding affinity, NO otherwise.
'''


class PairwiseProbRanker(Ranker):
    """
    Rank to maximize the sum of YES/NO probabilities between each pair.
    """
    def __init__(self, lm, sampler, description=TASK_DESCRIPTION, instruction=PAIRWISE_PROB_INSTRUCTION):
        self.lm = lm
        self.sampler = sampler
        self.description = description
        self.instruction = instruction

        self.answer_token_ids = {}
        for answer in ("YES", "NO"):
            token_ids = self.lm.tokenizer.encode(answer, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"{answer!r} must encode as a single token.")
            self.answer_token_ids[answer] = token_ids[0]


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
        """
        Return the YES probability normalized over YES and NO.
        """
        X_s, y_s = self.sampler.sample(ref=str_to_discrete([seq_a, seq_b]))
        samples = list(zip(discrete_to_str(X_s), y_s.tolist()))

        response = self.lm(self.format_prompt(samples, seq_a, seq_b), return_probs=True)
        if response.probs is None or response.probs.shape[0] == 0:
            raise ValueError("The language model returned no token probabilities.")

        yes_prob = response.probs[0, self.answer_token_ids["YES"]].item()
        no_prob = response.probs[0, self.answer_token_ids["NO"]].item()
        if yes_prob + no_prob <= 0:
            raise ValueError("YES and NO probabilities must have positive total mass.")
            
        return yes_prob / (yes_prob + no_prob)

    def rank(self, cands_str):
        """
        Ranking: x1, x2, x3, ... xn
        Greedily select the highest summed win probability against remaining candidates.
        """
        if len(cands_str) <= 1: return cands_str
    
        n = len(cands_str)
        predicted_ranking = [""]*n
        prob_matrix = torch.zeros((n, n), dtype=torch.float)

        for i in range(n):
            for j in range(n):
                if j == i: continue
                prob = self.pair_rank(cands_str[i], cands_str[j])
                prob_matrix[i][j] = prob

        prob_sum = torch.sum(prob_matrix, dim=1)
        remain_id = [i for i in range(n)]

        for i in range(n):
            current_id = remain_id[prob_sum[remain_id].argmax().item()]
            predicted_ranking[i] = cands_str[current_id]

            remain_id.remove(current_id)
            prob_sum[remain_id] -= prob_matrix[remain_id, current_id]

        return predicted_ranking
