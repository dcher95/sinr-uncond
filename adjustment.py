import numpy as np
import matplotlib.pyplot as plt 

## Bayesian-offset term
def bayesian_offset(conditioned_preds, unconditioned_preds, low_observation_density_threshold = 1):
    # Normalize unconditioned_preds to be between 0 and 1
    normalized_unconditioned = unconditioned_preds / unconditioned_preds.max()

    # Create a mask where unconditioned_preds is below the specified threshold
    mask = unconditioned_preds < low_observation_density_threshold

    # Apply the adjustment only where the mask is True
    adjusted_conditioned_preds = np.where(mask, conditioned_preds / normalized_unconditioned, conditioned_preds)

    # min-max normalize
    adjusted_conditioned_norm_preds = (adjusted_conditioned_preds - adjusted_conditioned_preds.min()) / (adjusted_conditioned_preds.max() - adjusted_conditioned_preds.min() + 1e-5)

    return adjusted_conditioned_norm_preds

## Bayesian-adjustment term
def bayesian_adjustment(conditioned_preds, unconditioned_preds, low_observation_density_threshold = 1):

    adjusted_numerator = conditioned_preds / unconditioned_preds
    adjusted_preds = adjusted_numerator / adjusted_numerator.sum()

    # Create a mask where unconditioned_preds is below the specified threshold
    mask = unconditioned_preds < low_observation_density_threshold

    # Apply the adjustment only where the mask is True
    adjusted_conditioned_preds = np.where(mask, adjusted_preds, conditioned_preds)

    # min-max normalize
    adjusted_conditioned_norm_preds = (adjusted_conditioned_preds - adjusted_conditioned_preds.min()) / (adjusted_conditioned_preds.max() - adjusted_conditioned_preds.min() + 1e-5)

    return adjusted_conditioned_norm_preds


## Logit-offset term
def logit_offset(inputs, densities, probs = False, alpha=1, epsilon=1e-5, centered=True, low_observation_density_threshold = 1):
    '''
    Assume inputs are logits. Convert probs to logits if necessary.
    '''
    if probs:
        # Convert probabilities to logits
        inputs = np.log(inputs / (1 - inputs + epsilon))  # Add epsilon to avoid division by zero

    # Calculate the mean log density
    log_densities = np.log(densities + epsilon)
    mean_log_density = np.mean(log_densities)

    #### Adjust logits only for low density areas
    # Calculate the percentile threshold for densities
    low_density_mask = densities < low_observation_density_threshold
    adjusted_logits = inputs.copy()
    
    # Adjust the logits using the centered log density
    if centered:
        adjusted_logits[low_density_mask] -= alpha * (log_densities[low_density_mask] - mean_log_density)
    else:
        adjusted_logits[low_density_mask] -= alpha * log_densities[low_density_mask]
    
    if probs:
        # Convert the adjusted logits back to probabilities
        return (1 / (1 + np.exp(-adjusted_logits)))  # Sigmoid function

    return adjusted_logits
