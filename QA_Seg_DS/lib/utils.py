import SimpleITK as sitk
import pickle
import numpy as np

def get_info_reader(path):
	reader = sitk.ImageFileReader()
	reader.SetFileName(path)
	reader.ReadImageInformation()
	reader.Execute()
	return reader

def read_image(path):
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)
    image = image[..., np.newaxis]
    return image
    
def read_mask(path, selected_labels):
    mask = read_image(path)

    result = np.zeros_like(mask)
    for new_lb, old_lb in enumerate(selected_labels):
        result[mask == old_lb] = new_lb + 1 # not include bg
    return result.astype(np.int)

def k_split(excel):
	df = pd.read_excel(excel, engine="openpyxl")
	sample_indices = list(range(len(df)))

	kf = KFold(K, shuffle=True)
	for i, (train_val_idxs, test_idxs) in enumerate(kf.split(sample_indices)):
		train_idxs, val_idxs = train_test_split(train_val_idxs, test_size=len(test_idxs), shuffle=True)
		train_df, val_df, test_df = df.iloc[train_idxs], df.iloc[val_idxs], df.iloc[test_idxs]

		folder = os.path.join(SAVE_DIR, "fold{}".format(i))
		os.mkdir(folder)
		train_df.to_excel(os.path.join(folder, "train.xlsx"), index=False)
		val_df.to_excel(os.path.join(folder, "val.xlsx"), index=False)
		test_df.to_excel(os.path.join(folder, "test.xlsx"), index=False)

def intensity_normalization(x, norm):
	if norm == "zscore":
		x = (x - x.mean()) / (x.std() + 1e-6)
	elif norm == "minmax":
		x = (x - x.min() / x.max() - x.min())
	return x