from typing import Optional

import torch
from torch.distributions.categorical import Categorical



class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None, device = None,softmax_annealing=False):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        
        self.mask_value = torch.tensor(torch.finfo(logits.dtype).min, dtype=logits.dtype).to(device)
        
        self.mask = self.mask.to(logits.dtype) 
        #first multiplication: sets logits to zero for impossible actions
        #secondly addition, keeps the possible actions prob. the same but adds -inf to the impossible ones
        logits = self.mask * logits + (1 - self.mask) * self.mask_value
        if softmax_annealing:
            #temperature paramter
            softmax_temp = 0.25
            logits_high_temp = torch.tensor([[x/softmax_temp for x in logits[0]]])
            super(CategoricalMasked, self).__init__(logits=logits_high_temp)
        else:
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        #point-wise addition/multiplication kernels are heavily 
        #optimized and being worked on. While the "where" kernel is not scrutinized as much
        p_log_p = self.logits * self.probs
        p_log_p *= self.mask
        
        return -p_log_p.sum(-1)
        """
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)
        """