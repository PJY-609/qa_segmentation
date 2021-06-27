import medpy.metric
import numpy as np


def measure_overlap(reference, prediction, overlap_metrics, n_labels):
	metrics = []
	for i in range(n_labels):
		ref = reference == (i + 1)
		pred = prediction == (i + 1)
		label_metrics = [getattr(medpy.metric, m)(ref, pred) for m in overlap_metrics]
		metrics.append(label_metrics)
	return metrics


def measure_surface(reference, prediction, surface_metrics, n_labels, spacing):
	metrics = []
	for i in range(n_labels):
		ref = reference == (i + 1)
		pred = prediction == (i + 1)
		label_metrics = [getattr(medpy.metric, m)(ref, pred, spacing) for m in surface_metrics]
		metrics.append(label_metrics)
	return metrics



def bootstrap_confidence_interval(data, n=1000, percentile=(2.5, 97.5)):
    simulations = []
    sample_size = len(data)

    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(itersample.mean())
    
    ci = (np.percentile(simulations, percentile[0]), np.percentile(simulations, percentile[1]))

    return ci