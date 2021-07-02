from lib.training.train import train


Config = {
	"data_config": {
		"k_train_excels": [
			"D:/QA/QA_Seg/preprocess/k_fold/fold0/train.xlsx"
			],
		"k_val_excels": [
			"D:/QA/QA_Seg/preprocess/k_fold/fold0/val.xlsx"
			],
		"image_header": "image",
		"mask_header": "scapula",
		"dmap_header": "scapula_dmap",
		"batch_size": 2,
		"n_patches_per_image": 6,
		"selected_labels":[1],
		"patch_size": (448, 448),
		"inten_norm": "zscore",
		"fg_ratio": 0.5,

		"num_points": 224 * 224,
		"oversample_factor": 4,
		"importance_sample_ratio": 0.7
	},

	"model_config": {
		"model": "unet",
		"n_channels": [1, 32, 64, 128, 256, 512],
		"norm": "instance",
		"nonlin": "lrelu",
		"dropout": 0.
	},

	"training_config": {
		"lr": 1e-3,
		"loss_weights": [1.],
		"plateau_patience": 3,
		"accum_grad": 6,
		"resume_training": "D:\\QA\\Quality_Assessment_Segmentation\\QA_Seg_DIR\\logs\\scapula\\version_5\\checkpoints\\last.ckpt",
		"experiment_name": "scapula",
		"log_dir": "logs",
		"earlystop_patience": 10
	}
}



if __name__ == "__main__":
	train(Config["model_config"], Config["training_config"], Config["data_config"])
		
