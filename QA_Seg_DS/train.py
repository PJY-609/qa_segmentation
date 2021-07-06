from lib.training.train import train

Config = {
	"data_config": {
		"k_train_excels": [
			"./preprocess/k_fold/fold0/train.xlsx"
			],
		"k_val_excels": [
			"./preprocess/k_fold/fold0/val.xlsx"
			],
		"image_header": "image",
		"mask_header": "scapula",
		"batch_size": 2,
		"n_patches_per_image": 6,
		"selected_labels":[1],
		"patch_size": (448, 448),
		"inten_norm": "zscore",
		"fg_ratio": 0.5
	},

	"model_config": {
		"model": "unet",
		"n_channels": [1, 32, 64, 128, 256, 512],
		"norm": "instance",
		"nonlin": "lrelu",
		"dropout": 0.
	},

	"training_config": {
		"optimizer": "Adam",
		"lr": 1e-3,
		"lr_schedule": {"scheduler": "plateau", "patience": 3},
		"loss_function": {"loss": "dc_and_ce", "weights": [1.]},
		
		"accum_grad": 6,
		"resume_training": "D:\\QA\\QA_Seg\\logs\\scapula_unet_deep_supervise\\version_0\\checkpoints\\last.ckpt",
		"experiment_name": "scapula_unet_deep_supervise",
		"log_dir": "logs",
		"earlystop_patience": 10
	}
}



if __name__ == "__main__":
	train(Config["model_config"], Config["training_config"], Config["data_config"])
