"""Evaluate incident classifiers on telemetry from separate synthetic site cohorts."""

import numpy as np
from scipy.special import expit, log_expit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_EPISODES_PER_SITE = 8
_SUPPORT_EVENTS = 10
_QUERY_EVENTS = 6

# Features are centered telemetry measurements: latency, CPU utilization,
# queue depth, packet loss, retry rate, and service errors.
# The recurring policies prioritize responsiveness, saturation, network health,
# and application reliability, respectively.
_POLICY_WEIGHTS = np.array([
    [3.0, 0.2, 0.5, 0.2, 0.2, 0.4],
    [0.2, 2.6, 1.4, 0.1, 0.1, 0.2],
    [0.3, 0.1, 0.1, 3.0, 1.1, 0.2],
    [0.2, 0.1, 0.2, 0.3, 0.7, 3.0],
])
_DEPLOYMENT_SHIFT = np.array([0.4, -0.2, 0.6, 0.7, 0.5, 0.1])
_BURST_EFFECT = np.array([0.7, 0.2, 0.7, 0.8, 0.9, 0.5])
_ASSIGNMENTS = ((np.arange(2**_QUERY_EVENTS)[:, None] >> np.arange(_QUERY_EVENTS)) & 1)


def _make_cohort(rng: np.random.Generator, sites: int, shift: float):
    """Draw new sites and generate episodes under their incident policies."""
    policy_count, feature_count = _POLICY_WEIGHTS.shape
    policies = np.arange(sites) % policy_count
    rng.shuffle(policies)
    events = _SUPPORT_EVENTS + _QUERY_EVENTS
    shape = (sites, _EPISODES_PER_SITE, events, feature_count)

    site_offsets = rng.normal(0.0, 0.3, size=(sites, 1, 1, feature_count))
    site_scales = rng.uniform(0.8, 1.25, size=(sites, 1, 1, feature_count))
    features = rng.normal(size=shape) * site_scales + site_offsets
    features += shift * _DEPLOYMENT_SHIFT
    features[..., 2] += 0.25 * features[..., 1]
    burst = rng.random(size=(*shape[:-1], 1)) < 0.12 + 0.17 * shift
    features += burst * _BURST_EFFECT

    site_thresholds = rng.normal(0.45, 0.3, size=(sites, 1, 1))
    logits = np.einsum("seif,sf->sei", features, _POLICY_WEIGHTS[policies]) - site_thresholds
    labels = rng.random(size=logits.shape) < expit(logits)
    return (
        features.reshape(-1, events, feature_count),
        labels.reshape(-1, events),
        np.repeat(policies, _EPISODES_PER_SITE),
    )


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=300)),
    ])
    return model.fit(features, labels)


def _predict_local(features: np.ndarray, support_labels: np.ndarray) -> np.ndarray:
    predictions = []
    for episode, labels in zip(features, support_labels):
        queries = episode[_SUPPORT_EVENTS:]
        if np.all(labels == labels[0]):
            predictions.append(np.full(_QUERY_EVENTS, labels[0]))
        else:
            local = _fit_logistic(episode[:_SUPPORT_EVENTS], labels)
            predictions.append(local.predict(queries))
    return np.asarray(predictions)


def _predict_rule_bank(
    bank: list[Pipeline], features: np.ndarray, support_labels: np.ndarray
) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    logits = np.stack([rule.decision_function(flat) for rule in bank])
    logits = logits.reshape(len(bank), *features.shape[:2])

    # Equal policy priors and independent labeled support observations give the
    # posterior over policies. Mix their 64 joint query assignments after that.
    signed_support = logits[..., :_SUPPORT_EVENTS] * np.where(support_labels, 1, -1)
    evidence = log_expit(signed_support).sum(axis=-1)
    log_policy = evidence - logsumexp(evidence, axis=0, keepdims=True)
    query_logits = logits[..., _SUPPORT_EVENTS:]
    log_assignments = query_logits @ _ASSIGNMENTS.T - np.logaddexp(0, query_logits).sum(
        axis=-1, keepdims=True
    )
    joint = logsumexp(log_policy[..., None] + log_assignments, axis=0)
    return _ASSIGNMENTS[joint.argmax(axis=-1)]


def _metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    correct = predictions == labels
    return {
        "accuracy": float(correct.mean()),
        "episode_exact_match": float(correct.all(axis=1).mean()),
    }


def evaluate(seed: int) -> dict[str, dict[str, dict[str, float]]]:
    """Fit the global and policy models once and score both independent cohorts."""
    train_rng, validation_rng, test_rng = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(3))
    features, labels, policies = _make_cohort(train_rng, sites=40, shift=0.0)
    dimension = features.shape[-1]
    pooled = _fit_logistic(features.reshape(-1, dimension), labels.ravel())
    bank = []
    for policy in range(len(_POLICY_WEIGHTS)):
        rows = policies == policy
        bank.append(_fit_logistic(features[rows].reshape(-1, dimension), labels[rows].ravel()))

    scores = {}
    for name, rng, sites, shift in (
        ("validation", validation_rng, 16, 0.3),
        ("test", test_rng, 20, 0.7),
    ):
        features, labels, _ = _make_cohort(rng, sites, shift)
        support_labels = labels[:, :_SUPPORT_EVENTS]
        queries = features[:, _SUPPORT_EVENTS:]
        query_labels = labels[:, _SUPPORT_EVENTS:]
        pooled_predictions = pooled.predict(queries.reshape(-1, dimension)).reshape(query_labels.shape)
        scores[name] = {
            "global": _metrics(pooled_predictions, query_labels),
            "local": _metrics(_predict_local(features, support_labels), query_labels),
            "rule_bank": _metrics(_predict_rule_bank(bank, features, support_labels), query_labels),
        }
    return scores
