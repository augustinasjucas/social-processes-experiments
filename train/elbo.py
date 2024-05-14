#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: elbo.py
# Created Date: Saturday, November 9th 2019, 1:11:59 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.kl import kl_divergence

from data.types import Seq2SeqPredictions


def log_likelihood(normal: torch.distributions.Normal, target: Tensor) -> Tensor:
    """ Compute the log probability at the target given a normal distribution

    Take mean over the z_samples, target_len, and nsequences, and sum over
    the rest of the dimensions

    Args:
        normal  : The normal predicted distribution with parameter shape
                  (nz_samples, target_len, nsequences, ...)
        target  : The target at which to compute the log probability
                  (target_len, nsequences, ...)

    Returns the log probability evaluated at the target

    """
    return normal.log_prob(target).mean(dim=(0, 1, 2)).sum()


class SocialProcessSeq2SeqElbo(nn.Module):

    """ Compute the negative of elbo term for the neural process """

    def forward(
            self, pred: Seq2SeqPredictions, target_future: torch.Tensor
        ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """ Compute the ELBO.
        Args:
            pred            : The artefacts of the forward pass of the
                              SocialProcessSeq2Seq model
            target_future   : The ground truth predictions of the target
                              sequences

        Returns the overall Elbo and the constituent NLL and KL terms

        """
        qs = pred.posteriors

        # In case of a mixture model, the pred.stochastic is a tuple
        if isinstance(pred.stochastic, tuple):
            # The pred.stochastic is a tuple of (normal_1, normal_2, k)
            normal_1 = pred.stochastic[0]
            normal_2 = pred.stochastic[1]
            k = pred.stochastic[2].view([])
            eps = torch.tensor(1e-10)
            # Calculate the nll for the mixture model, using logsumexp
            loss = -torch.logsumexp(torch.stack([torch.log(k + eps) + log_likelihood(normal_1, target_future), torch.log(1-k + eps) + log_likelihood(normal_2, target_future)], dim=0), dim=0)

        # In the default case of a single normal distribution, just calculate the nll as usual
        else:
            loss = - log_likelihood(pred.stochastic, target_future)

        nll = loss.detach().clone() # important for preserving value
        kl = None
        if qs.q_target is not None:
            # Take the mean over the batch size and sum over the last dim
            # The q distrib tensor dims are (batch_size, z_dim)
            kl = kl_divergence(qs.q_target, qs.q_context).mean(dim=0).sum()
            loss += kl
            kl = kl.detach().clone() # important for preserving value
        return loss, nll, kl
