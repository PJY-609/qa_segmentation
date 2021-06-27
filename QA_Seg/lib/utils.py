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
	
    if image.ndim == 2:
    	image = image[:, :, np.newaxis]
    elif image.ndim == 3:
    	image = np.moveaxis(image, 0, -1)

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


def replace_pixval(image, pixvals):
	POS = np.isin(image, pixvals)
	image[POS] = image[~POS].mean()
	return image


def resample(image, new_spacing, interpolator):
	old_shape, old_spacing = image.GetSize(), image.GetSpacing()
	new_spacing = (new_spacing[0], new_spacing[1], 1.) if len(old_spacing) > len(new_spacing) else new_spacing

	new_shape = np.ceil(np.multiply(old_shape, np.divide(old_spacing, new_spacing))).astype(np.int32).tolist()

	resample_filter = sitk.ResampleImageFilter()
	resample_filter.SetInterpolator(interpolator)
	resample_filter.SetSize(new_shape)
	resample_filter.SetOutputSpacing(new_spacing)
	resample_filter.SetOutputDirection(image.GetDirection())
	resample_filter.SetOutputOrigin(image.GetOrigin())

	image = resample_filter.Execute(image)
	return image



def whd_2_hwd(whd):
	w, h, d = whd
	return (h, w, d)
