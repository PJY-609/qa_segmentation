


def metric_logging(log, metric_names, mask_type, labels, mask_metrics):
	for i, metrics in enumerate(mask_metrics):
		log.update({
			"_".join((metric_names[i], mask_type, str(labels[i]))): m for m in metrics
			})
            