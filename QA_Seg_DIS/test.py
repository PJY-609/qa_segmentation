import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from lib.render_module import RenderModule



Config = {
		"k_checkpoints": [
			"D:\\QA\\Quality_Assessment_Segmentation\\QA_Seg_DIR\\logs\\scapula\\version_4\\checkpoints\\epoch=8-val_loss=0.6109.ckpt",
			# "D:\\QA\\Quality_Assessment_Segmentation\\QA_Seg_MTL\\logs\\scapula\\version_2\\checkpoints\\epoch=79-val_loss=0.0704.ckpt",
			# "D:\\QA\\Quality_Assessment_Segmentation\\QA_Seg_MTL\\logs\\scapula\\version_3\\checkpoints\\epoch=59-val_loss=0.0760.ckpt",
			# "D:\\QA\\Quality_Assessment_Segmentation\\QA_Seg_MTL\\logs\\scapula\\version_4\\checkpoints\\epoch=66-val_loss=0.0827.ckpt",
			# "D:\\QA\\Quality_Assessment_Segmentation\\QA_Seg_MTL\\logs\\scapula\\version_5\\checkpoints\\epoch=67-val_loss=0.0748.ckpt",
			# "D:\\QA\\Quality_Assessment_Segmentation\\QA_Seg_MTL\\logs\\scapula\\version_6\\checkpoints\\epoch=58-val_loss=0.0690.ckpt",
		
		],

		"k_test_excels": [
			"D:\\QA\\Quality_Assessment_Segmentation\\k_fold\\fold0\\test.xlsx",
			# "D:\\QA\\Quality_Assessment_Segmentation\\k_fold\\fold1\\test.xlsx",
			# "D:\\QA\\Quality_Assessment_Segmentation\\k_fold\\fold2\\test.xlsx",
			# "D:\\QA\\Quality_Assessment_Segmentation\\k_fold\\fold3\\test.xlsx",
			# "D:\\QA\\Quality_Assessment_Segmentation\\k_fold\\fold4\\test.xlsx",
			# "D:\\QA\\Quality_Assessment_Segmentation\\k_fold\\fold5\\test.xlsx",
			# "D:\\QA\\Quality_Assessment_Segmentation\\k_fold\\fold6\\test.xlsx",
			# "D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold7\\test.xlsx",
			# 
			# "D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold8\\test.xlsx",
			# "D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold9\\test.xlsx"
		],
		

        "batch_size": 2,
        "n_largest_components": [2],
		
		"image_header": "org_image",
		"mask_header": "org_scapula",
		"selected_labels": [1],

		"pixval_replacement": [16383],
		"new_spacing": (0.5, 0.5),
		"patch_size": (448, 448),
		"normalization": "zscore",

		"overlap_metrics": ["dc"],
		"surface_metrics": ["assd"],

		"log_dir": "result_evaluation",
		"experiment_name": "scapula"
}


def main():
	Config.setdefault("test_excel")

	for ckpt, excel in zip(Config["k_checkpoints"], Config["k_test_excels"]):
		Config["test_excel"] = excel
		test_module = RenderModule.load_from_checkpoint(ckpt, test_config=Config)

		logger = CSVLogger(Config["log_dir"], name=Config["experiment_name"])
		trainer = pl.Trainer(gpus=1, logger=logger)
		trainer.test(test_module)


if __name__ == '__main__':
	main()
