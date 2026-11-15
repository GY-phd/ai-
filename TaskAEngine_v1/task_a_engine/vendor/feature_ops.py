"""Generated from the tested experiment feature functions."""
import numpy as np
GROUP = np.tile(np.repeat(np.arange(4), 2), 2)
CLASSES = ['Ethanol', 'Ethylene', 'Ammonia', 'Acetaldehyde', 'Acetone', 'Toluene']

def transform(x, view):
    s = x.reshape(-1, 16, 8)
    if view == 'raw':
        return x
    if view == 'log':
        return np.sign(x) * np.log1p(np.abs(x))
    steady = np.sign(s[:, :, 0]) * np.log1p(np.abs(s[:, :, 0]))
    relative_delta = s[:, :, 1] - 1
    relative = np.sign(relative_delta) * np.log1p(np.abs(relative_delta))
    steady -= steady.mean(axis=1, keepdims=True)
    relative -= relative.mean(axis=1, keepdims=True)
    ema = np.log(np.maximum(np.abs(s[:, :, 2:]), 0.001))
    log_amplitude = np.log(np.maximum(np.abs(s[:, :, :1]), 0.001))
    rising_over_steady = (ema[:, :, :3] - log_amplitude).reshape(len(x), 48)
    kinetic = np.concatenate((ema[:, :, 1:3] - ema[:, :, :1], ema[:, :, 4:6] - ema[:, :, 3:4], ema[:, :, 3:4] - ema[:, :, :1]), axis=2).reshape(len(x), 80)
    if view == 'steady_profile':
        return np.column_stack((steady, relative))
    if view == 'kinetics':
        return kinetic
    if view == 'rising_over_steady':
        return rising_over_steady
    if view == 'profile_kinetics':
        return np.column_stack((steady, relative, rising_over_steady))
    ema -= ema.mean(axis=1, keepdims=True)
    return np.column_stack((steady, relative, ema.reshape(len(x), 96)))

def softmax(scores):
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

def score(expected, probabilities):
    labels = probabilities.argmax(axis=1)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(matrix, (expected, labels), 1)
    correct, support, predicted = (matrix.diagonal(), matrix.sum(axis=1), matrix.sum(axis=0))
    per_class = {}
    for i, name in enumerate(CLASSES):
        per_class[name] = {'support': int(support[i]), 'correct': int(correct[i]), 'predicted': int(predicted[i]), 'precision': float(correct[i] / max(1, predicted[i])), 'recall': float(correct[i] / max(1, support[i])), 'f1': float(2 * correct[i] / max(1, support[i] + predicted[i]))}
    return {'macro_f1': float(np.mean([v['f1'] for v in per_class.values()])), 'accuracy': float(correct.sum() / support.sum()), 'per_class': per_class, 'worst_recall': min((v['recall'] for v in per_class.values() if v['support'])), 'confusion_matrix': matrix.tolist()}

def project(values, by_type):
    """Remove common log EMA slopes separately for rising and decaying response."""
    z = values.reshape(-1, 16, 5).copy()
    z[:, :, 4] += z[:, :, 2] - z[:, :, 0]
    slopes = np.array([1.0, 2.0])
    groups = [np.arange(16)] if not by_type else [np.flatnonzero(GROUP == i) for i in range(4)]
    for group in groups:
        part = z[:, group, :4].reshape(len(z), len(group), 2, 2)
        slope = (part @ slopes).mean(axis=1, keepdims=True) / 5
        part -= slope[..., None] * slopes
        z[:, group, :4] = part.reshape(len(z), len(group), 4)
    return z.reshape(len(z), 80)
