import pytorch_lightning as pl

from lib.segment_module import SegmentModule



Config = {
		"k_checkpoints": [
			"D:\\QA\\Quality_Assessment_Segmentation\\QA_Seg_DS\\logs\\scapula_ds\\version_0\\checkpoints\\epoch=81-val_loss=0.1061.ckpt"
		],
		"k_test_excels": [
		"D:/QA/QA_Seg/preprocess/k_fold/fold0/test.xlsx"
		],
		"batch_size": 2,
		"patch_size": (448, 448),
		"normalization": "zscore",
		"k_result_dirs": [
		"D:\\QA\\Quality_Assessment_Segmentation\\QA_Seg_DS\\results"
		]
}


def main():
	Config.setdefault("test_excel")
	Config.setdefault("result_dir")

	for ckpt, excel, dir_ in zip(Config["k_checkpoints"], Config["k_test_excels"], Config["k_result_dirs"]):
		Config["test_excel"], Config["result_dir"] = excel, dir_
		test_module = SegmentModule.load_from_checkpoint(ckpt, test_config=Config)
		trainer = pl.Trainer(gpus=1, logger=False)
		trainer.test(test_module)


if __name__ == '__main__':
	main()
