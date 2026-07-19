"""
BTWM: Bidirectional Transition World Model
Inverse action prediction head for DreamerV3's RSSM.
Predicts p(a_t | z_t, z_{t+1}) from consecutive latent states.

Supports both discrete (Atari/Crafter) and continuous (DMC) action spaces.
Continuous actions are discretized into bins (default 21).
"""
import jax
import jax.numpy as jnp
import ninjax as nj
import embodied.jax.nets as nn
import numpy as np

f32 = jnp.float32


def discrete_classes(space):
    """Return the number of classes for a scalar discrete action space."""
    classes = np.asarray(space.classes)
    if space.shape or classes.shape:
        raise ValueError(
            f"BTWM only supports scalar discrete actions, got shape={space.shape}")
    return int(classes.item())


def action_labels(actions, space, num_bins=5):
    """Convert raw environment actions to zero-based classifier labels."""
    if space.discrete:
        low = np.asarray(space.low)
        if space.shape or low.shape:
            raise ValueError(
                f"BTWM only supports scalar discrete actions, got shape={space.shape}")
        return jnp.asarray(actions, jnp.int32) - int(low.item())

    low = np.asarray(space.low, np.float32)
    high = np.asarray(space.high, np.float32)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("Continuous BTWM actions require finite bounds.")
    if np.any(high <= low):
        raise ValueError(f"Invalid continuous action bounds: low={low}, high={high}")

    actions = jnp.asarray(actions, f32)
    scaled = (actions - jnp.asarray(low)) / jnp.asarray(high - low)
    labels = jnp.floor(scaled * num_bins).astype(jnp.int32)
    return jnp.clip(labels, 0, num_bins - 1)


def masked_mean(values, mask):
    """Mean over valid transitions, returning zero when none are valid."""
    values = jnp.asarray(values, f32)
    mask = jnp.asarray(mask, bool)
    mask = jnp.broadcast_to(mask, values.shape)
    count = mask.sum()
    total = jnp.where(mask, values, 0).sum()
    return jnp.where(count > 0, total / jnp.maximum(count, 1), 0.0)


def inverse_confidence(inv_logits, actions, act_space):
    """Geometric mean probability assigned to the true action."""
    ce = inverse_loss(inv_logits, actions, act_space, normalize=False)
    return jnp.exp(-ce)


def relative_confidence_weights(confidence, minimum=0.5, maximum=1.5):
    """Convert confidence values to bounded, unit-mean sample weights."""
    confidence = jnp.asarray(confidence, f32)
    mean = confidence.mean()
    relative = confidence / jnp.maximum(mean, 1e-6)
    weights = jnp.clip(relative, minimum, maximum)
    return weights / jnp.maximum(weights.mean(), 1e-6)


def transition_targets(prev_actions, reset):
    """Return actions causing z_t -> z_{t+1} and their validity mask."""
    targets = {key: value[:, 1:] for key, value in prev_actions.items()}
    valid = ~reset[:, 1:]
    return targets, valid


class InverseActionHead(nj.Module):
    """MLP that predicts action from two consecutive feature vectors.

    Input: x_t, x_{t+1}, each shaped (B, T-1, D).
    Output: dict of logits, one per action subspace key.
    """

    hidden_dim: int = 256
    depth: int = 2
    norm: str = 'rms'
    act: str = 'gelu'
    num_bins: int = 21

    def __init__(self, act_space, **kw):
        self.act_space = act_space
        self.kw = kw

    def __call__(self, x_t, x_next):
        x = jnp.concatenate([x_t, x_next], axis=-1)

        for i in range(self.depth):
            x = self.sub(f'lin{i}', nn.Linear, self.hidden_dim, **self.kw)(x)
            x = nn.act(self.act)(
                self.sub(f'norm{i}', nn.Norm, self.norm)(x))

        logits = {}
        for k, space in self.act_space.items():
            if space.discrete:
                # Discrete action: output (B, T-1, num_actions)
                num_actions = discrete_classes(space)
                logits[k] = self.sub(
                    f'out_{k}', nn.Linear, num_actions, **self.kw)(x)
            else:
                # Continuous action: discretize into bins
                num_actions = int(np.prod(space.shape))
                out_dim = num_actions * self.num_bins
                logits[k] = self.sub(
                    f'out_{k}', nn.Linear, out_dim, **self.kw)(x)
                logits[k] = logits[k].reshape(
                    (*logits[k].shape[:-1], num_actions, self.num_bins))
        return logits


def inverse_loss(inv_logits, actions, act_space, normalize=False):
    """Per-element cross-entropy inverse loss.

    Args:
        inv_logits: {key: (B, T-1, A) or (B, T-1, A, C)} logits dict
        actions: {key: (B, T-1) or (B, T-1, A)} raw action dict
        act_space: action space

    Returns:
        (B, T-1) loss tensor, averaged over action keys. If normalize is true,
        each action key is divided by its random-policy cross entropy so that
        an untrained classifier has loss approximately one.
    """
    total = None
    n = 0
    for k in act_space.keys():
        logits = inv_logits[k]
        space = act_space[k]
        labels = action_labels(actions[k], space, logits.shape[-1])
        if not space.discrete:
            # Continuous: logits (B,T,A,C), labels (B,T,A)
            # Average over action dimensions to keep the scale task-independent.
            onehot = jax.nn.one_hot(labels, logits.shape[-1])
            ce = -jnp.sum(
                onehot * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            ce = ce.mean(axis=-1)
        else:
            # Discrete: logits (B,T,C), labels (B,T)
            # one_hot gives (B,T,C), sum CE over class dim -> (B,T)
            ce = -jnp.sum(
                jax.nn.one_hot(labels, logits.shape[-1]) *
                jax.nn.log_softmax(logits, axis=-1),
                axis=-1)
        if normalize:
            ce /= jnp.log(jnp.asarray(logits.shape[-1], f32))
        total = ce if total is None else total + ce
        n += 1
    return total / max(n, 1)


def inverse_accuracy(inv_logits, actions, act_space):
    """Per-transition inverse prediction accuracy."""
    acc = None
    n = 0
    for k in act_space.keys():
        logits = inv_logits[k]
        space = act_space[k]
        labels = action_labels(actions[k], space, logits.shape[-1])
        preds = jnp.argmax(logits, axis=-1)
        if space.discrete:
            correct = (preds == labels).astype(f32)
        else:
            correct = (preds == labels).astype(f32).mean(axis=-1)
        acc = correct if acc is None else acc + correct
        n += 1
    return acc / max(n, 1)
