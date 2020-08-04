import numpy as np


class Hypothesis(object):
    def __init__(self, tokens, log_probs, state):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state

    def extend(self, token, log_prob, state):

        return Hypothesis(tokens=self.tokens + [token],
                          log_probs=self.log_probs + [log_prob],
                          state=state)

    @property
    def last_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.log_prob / len(self.tokens)


def sort_hyps(hyps):

    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)


def beam_search(sess, model, vocab, source, source_len):
    beam_size = model.beam_size
    source = np.tile(source, [beam_size, 1])
    source_len = np.tile(source_len, beam_size)
    encoder_outputs, dec_inp_state = model.encoder_run(sess, source, source_len)

    hyps = [Hypothesis(tokens=[vocab.start], log_probs=[0.0], state=dec_inp_state)
            for _ in range(beam_size)]

    results = []
    step = 0
    while step < model.max_decode_step and len(results) < beam_size:
        last_tokens = [h.last_token for h in hyps]  # list of scalar
        states = [h.state for h in hyps]

        topk_idx, topk_log_probs, new_states = model.decode_onestep(sess, last_tokens,
                                                     states, encoder_outputs, source_len)
        all_hyps = []
        num_hyps = 1 if step == 0 else len(hyps)
        for i in range(num_hyps):
            hyps_, new_state = hyps[i], new_states[i]
            for j in range(beam_size * 2):
                temp = hyps_.extend(token=topk_idx[i, j],
                                    log_prob=topk_log_probs[i, j],
                                    state=new_state)
                all_hyps.append(temp)

        hyps = []
        for hyps_ in sort_hyps(all_hyps):
            if hyps_.last_token == vocab.end:
                if step >= 1:
                    results.append(hyps_)
            else:
                hyps.append(hyps_)
            if len(hyps) == beam_size or len(results) == beam_size:
                break

        step += 1

    if len(results) == 0:
        results = hyps

    return sort_hyps(results)[0].tokens
