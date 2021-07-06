import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from lib.segment_module import SegmentModule



Config = {
		"k_checkpoints": [
			"D:\\Workspace10\\Qualify_Assessment_Segmentation\\QA_Seg_DS\\logs\\clavicle\\version_0\\checkpoints\\epoch=41-val_loss=0.0854.ckpt",
			"D:\\Workspace10\\Qualify_Assessment_Segmentation\\QA_Seg_DS\\logs\\clavicle\\version_1\\checkpoints\\epoch=35-val_loss=0.0886.ckpt",
			"D:\\Workspace10\\Qualify_Assessment_Segmentation\\QA_Seg_DS\\logs\\clavicle\\version_2\\checkpoints\\epoch=61-val_loss=0.0822.ckpt",
			"D:\\Workspace10\\Qualify_Assessment_Segmentation\\QA_Seg_DS\\logs\\clavicle\\version_3\\checkpoints\\epoch=54-val_loss=0.0796.ckpt",
			"D:\\Workspace10\\Qualify_Assessment_Segmentation\\QA_Seg_DS\\logs\\clavicle\\version_4\\checkpoints\\epoch=56-val_loss=0.0785.ckpt",
			"D:\\Workspace10\\Qualify_Assessment_Segmentation\\QA_Seg_DS\\logs\\clavicle\\version_5\\checkpoints\\epoch=48-val_loss=0.0847.ckpt",
		],

		"k_test_excels": [
			"D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold0\\test.xlsx",
			"D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold1\\test.xlsx",
			"D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold2\\test.xlsx",
			"D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold3\\test.xlsx",
			"D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold4\\test.xlsx",
			"D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold5\\test.xlsx",
			# "D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold6\\test.xlsx",
			# "D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold7\\test.xlsx",
			# "D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold8\\test.xlsx",
			# "D:\\Workspace10\\QA_Seg\\preprocess\\k_fold\\fold9\\test.xlsx"
		],
		

        "batch_size": 2,
        "n_largest_components": [2],
		
		"image_header": "org_image",
		"mask_header": "org_clavicle",
		"selected_labels": [1],

		"pixval_replacement": [16383],
		"new_spacing": (0.5, 0.5),
		"patch_size": (448, 448),
		"normalization": "zscore",

		"overlap_metrics": ["dc"],
		"surface_metrics": ["assd"],

		"log_dir": "result_evaluation",
		"experiment_name": "clavicle"
}


def main():
	Config.setdefault("test_excel")

	for ckpt, excel in zip(Config["k_checkpoints"], Config["k_test_excels"]):
		Config["test_excel"] = excel
		test_module = SegmentModule.load_from_checkpoint(ckpt, test_config=Config)

		logger = CSVLogger(Config["log_dir"], name=Config["experiment_name"])
		trainer = pl.Trainer(gpus=1, logger=logger)
		trainer.test(test_module)


if __name__ == '__main__':
	main()
