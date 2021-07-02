import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from pathlib import Path
from itertools import chain

from lib.render_module import RenderModule

import matplotlib.pyplot as plt


def train_one_fold(model_config, training_config, data_config):
	module = RenderModule(model_config, training_config, data_config)

	lr_monitor = LearningRateMonitor()
	early_stopping = EarlyStopping(monitor='val_loss', patience=training_config["earlystop_patience"], verbose=True, mode='min')
	model_checkpoint = ModelCheckpoint(monitor="val_loss", filename='{epoch}-{val_loss:.4f}', save_last=True, save_top_k=1, save_weights_only=False, mode='min')
	callbacks = [model_checkpoint, early_stopping, lr_monitor]

	logger = CSVLogger(training_config["log_dir"], name=training_config["experiment_name"])
	trainer = pl.Trainer(resume_from_checkpoint=training_config["resume_training"], gpus=1, logger=logger, max_epochs=1000, accumulate_grad_batches=training_config["accum_grad"], callbacks=callbacks)
	trainer.fit(module)

def train(model_config, training_config, data_config):
	data_config.setdefault("train_excel")
	data_config.setdefault("val_excel")

	for te, ve in zip(data_config["k_train_excels"], data_config["k_val_excels"]):
		data_config["train_excel"] = te
		data_config["val_excel"] = ve
		train_one_fold(model_config, training_config, data_config)

