import unittest
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = ROOT / "dreamerv3"
if str(DREAMERV3_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMERV3_ROOT))

import elements
import jax.numpy as jnp
import numpy as np

from code.btwm_head import (
    action_labels,
    discrete_classes,
    inverse_accuracy,
    inverse_confidence,
    inverse_loss,
    masked_mean,
    relative_confidence_weights,
    transition_targets,
)


class BtwmHeadTest(unittest.TestCase):

    def test_discrete_space_uses_exclusive_high_bound(self):
        space = elements.Space(np.int32, (), 2, 7)
        self.assertEqual(discrete_classes(space), 5)
        labels = action_labels(jnp.array([[2, 6]]), space)
        np.testing.assert_array_equal(labels, [[0, 4]])

    def test_discrete_inverse_loss_and_accuracy(self):
        space = elements.Space(np.int32, (), 0, 3)
        actions = {'action': jnp.array([[0, 2]], jnp.int32)}
        logits = {'action': jnp.array([[
            [10.0, -10.0, -10.0],
            [-10.0, -10.0, 10.0],
        ]])}
        spaces = {'action': space}

        loss = inverse_loss(logits, actions, spaces)
        accuracy = inverse_accuracy(logits, actions, spaces)

        self.assertEqual(loss.shape, (1, 2))
        np.testing.assert_allclose(accuracy, 1.0)
        self.assertLess(float(loss.max()), 1e-6)

    def test_continuous_actions_are_discretized(self):
        space = elements.Space(np.float32, (5,), -1.0, 1.0)
        actions = jnp.array([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        labels = action_labels(actions, space, num_bins=5)
        np.testing.assert_array_equal(labels, [[0, 1, 2, 3, 4]])

    def test_continuous_loss_averages_action_dimensions(self):
        space = elements.Space(np.float32, (2,), -1.0, 1.0)
        actions = {'action': jnp.array([[[-1.0, 1.0], [0.0, 0.5]]])}
        logits = {'action': jnp.zeros((1, 2, 2, 5))}
        spaces = {'action': space}

        loss = inverse_loss(logits, actions, spaces)

        np.testing.assert_allclose(loss, np.log(5.0), rtol=1e-6)

    def test_normalized_uniform_loss_is_one(self):
        space = elements.Space(np.float32, (2,), -1.0, 1.0)
        actions = {'action': jnp.zeros((1, 3, 2))}
        logits = {'action': jnp.zeros((1, 3, 2, 21))}

        loss = inverse_loss(
            logits, actions, {'action': space}, normalize=True)

        np.testing.assert_allclose(loss, 1.0, rtol=1e-6)

    def test_inverse_confidence_is_true_action_probability(self):
        space = elements.Space(np.int32, (), 0, 2)
        actions = {'action': jnp.array([[0]], jnp.int32)}
        logits = {'action': jnp.log(jnp.array([[[0.8, 0.2]]]))}

        confidence = inverse_confidence(logits, actions, {'action': space})

        np.testing.assert_allclose(confidence, 0.8, rtol=1e-6)

    def test_relative_confidence_weights_are_bounded_and_unit_mean(self):
        weights = relative_confidence_weights(
            jnp.array([0.01, 0.1, 1.0]), minimum=0.5, maximum=1.5)

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights[-1]), float(weights[0]))

    def test_transition_targets_shift_and_mask_episode_boundaries(self):
        prev_actions = {'action': jnp.array([[10, 11, 12, 13]])}
        reset = jnp.array([[True, False, True, False]])

        targets, valid = transition_targets(prev_actions, reset)

        np.testing.assert_array_equal(targets['action'], [[11, 12, 13]])
        np.testing.assert_array_equal(valid, [[True, False, True]])

    def test_masked_mean_ignores_invalid_transitions(self):
        values = jnp.array([[1.0, 100.0, 3.0]])
        valid = jnp.array([[True, False, True]])
        self.assertAlmostEqual(float(masked_mean(values, valid)), 2.0)
        self.assertEqual(
            float(masked_mean(values, jnp.zeros_like(valid))), 0.0)


if __name__ == '__main__':
    unittest.main()
