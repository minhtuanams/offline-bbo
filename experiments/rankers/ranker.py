class Ranker():
    def __init__(self, lm):
        self.lm = lm

    def rank(self, samples, cands_str):
        '''
        Return a list of cands_str sorted from highest to lowest score
        '''