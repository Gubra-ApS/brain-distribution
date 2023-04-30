import numpy as np
import pandas as pd
import SimpleITK as sitk

from skimage.io import imread
from skimage.transform import rescale, resize
from scipy.ndimage import labeled_comprehension


'''
### Function definitions
'''
def load_nifti(file_name):
    """
    Load a NIfTI file and return its voxel data as a numpy array.

    Parameters:
    file_name (str): Path to the NIfTI file.

    Returns:
    volume (numpy array): The voxel data of the NIfTI file as a 3D numpy array.
    """
    volume = sitk.GetArrayFromImage(sitk.ReadImage(file_name))
    return volume


def save_nifti(vol, file_name, voxel_size=[1, 1, 1]):
    """
    Save a 3D numpy array as a NIfTI file.

    Parameters:
    vol (numpy array): A 3D numpy array of voxel data.
    file_name (str): Path to the output NIfTI file.
    voxel_size (list, optional): Voxel size of the output NIfTI file. Default is [1, 1, 1].

    Returns:
    None
    """
    vol_sitk = sitk.GetImageFromArray(vol)
    vol_sitk.SetSpacing((float(voxel_size[2]), float(voxel_size[1]), float(voxel_size[0])))
    vol_sitk.SetOrigin((round(vol_sitk.GetWidth()/2), round(vol_sitk.GetHeight()/2), -round(vol_sitk.GetDepth()/2)))
    sitk.WriteImage(vol_sitk, file_name)


def downsample_volume(vol, voxel_size_in, voxel_size_out, interp_order=1, dtype_out='uint16'):
    """
    Downsample a 3D numpy array by a given factor.

    Parameters:
    vol (numpy array): A 3D numpy array of voxel data.
    voxel_size_in (list): Voxel size of the input numpy array.
    voxel_size_out (list): Voxel size of the output numpy array.
    interp_order (int, optional): The order of interpolation used. Default is 1.
    dtype_out (str, optional): Data type of the output numpy array. Default is 'uint16'.

    Returns:
    vol_out (numpy array): The downsampled 3D numpy array of voxel data.
    """

    # first, downsample xy-direction slice-by-slice
    vol_out = []
    scale_xy = np.array(voxel_size_in[1:]) / voxel_size_out
    for z in range(vol.shape[0]):
        vol_out.append(rescale(vol[z], scale=scale_xy, order=interp_order, preserve_range=True).astype(dtype_out))
    vol_out = np.array(vol_out)

    # next downsample z-direction and return
    scale_z = np.array([voxel_size_in[0]/voxel_size_out, 1, 1])
    return rescale(vol_out, scale=scale_z, order=interp_order, preserve_range=True).astype(dtype_out)


def upsample_volume(vol, shape_out, interp_order=1):
    """
    Upsample a 3D numpy array to a given shape.

    Parameters:
    vol (numpy array): A 3D numpy array of voxel data.
    shape_out (tuple): Desired output shape of the numpy array.
    interp_order (int, optional): The order of interpolation used. Default is 1.

    Returns:
    vol_out (numpy array): The upsampled 3D numpy array of voxel data.
    """

    # cast everything to 'float32' and init output volume
    vol_dtype = vol.dtype
    vol = vol.astype('float32')
    vol_out = np.zeros((vol.shape[0], shape_out[1], shape_out[2]), dtype=vol_dtype)

    # upsample in xy
    for z in range(vol.shape[0]):
        vol_out[z] = resize(vol[z], (shape_out[1], shape_out[2]), order=interp_order, preserve_range=True)

    # upsample in z and return
    vol_out = resize(vol_out, shape_out, order=interp_order, preserve_range=True)
    return vol_out.astype(vol_dtype)


def unmix(auto, spec, nbins=32):
    """
    Unmix autofluorescence from specific signal using a histogram-based method.

    Parameters:
    auto (numpy array): A 3D numpy array of autofluorescence data.
    spec (numpy array): A 3D numpy array of spectral data.
    nbins (int, optional): Number of bins to use in the histogram. Default is 32.

    Returns:
    unmix (numpy array): The unmixed spectral data as a 3D numpy array.
    """

    # create histogram
    h0 = np.histogram(auto.flatten(), bins=nbins, range=(0, np.percentile(auto, 99.99)))

    # calculate ratio between all voxels represented within each histogram bin
    ratio = np.zeros((nbins, 1))
    estimate = auto.copy().astype('float32')

    for k in range(nbins):
        print(k + 1, '/', nbins)
        idx = (auto > h0[1][k]) & (auto < h0[1][k + 1])
        v0 = auto[idx]
        v1 = spec[idx]

        if (len(v0) > 0) and (len(v1) > 0):
            ratio[k] = np.nanmedian(np.divide(v0, v1))
            estimate[idx] = estimate[idx] / ratio[k]

    # unmix by subtracting estimated autofluorescence
    unmix = spec.astype('float32') - estimate.astype('float32')
    unmix[unmix < 0] = 0
    return unmix.astype('uint16')


def region_wise_quantification(signal, atlas_anno, region_ids):
    """
    Calculates the region-wise quantification of volume intensities based on the provided signal and atlas annotations.

    Args:
        signal (numpy.ndarray): A 3D volume containing the signal to quantify.
        atlas_anno (numpy.ndarray): A 3D volume containing the atlas annotations.
        region_ids (list): A list of integer IDs corresponding to the regions of interest in the atlas.

    Returns:
        numpy.ndarray: An array containing the region-wise quantification of volume intensities for the specified
        regions of interest.
    """
    return labeled_comprehension(signal, atlas_anno, region_ids, np.sum, float, 0)




'''
### Run unmixing example script
'''
if __name__ == '__main__':

    # input files
    input_auto_image = '/input/auto.tif'  # autofluorescence
    input_spec_image = '/input/spec.tif'  # specific channel
    input_atlas_template = '/input/atlas_template.nii.gz'
    input_atlas_annotations = '/input/atlas_annotations.nii.gz'
    input_annotation_template = '/input/annotation_template.csv'

    # output files
    output_downsampled_auto = '/output/auto_downsampled.nii.gz'
    output_spec_unmix_mapped = '/output/spec_unmix_mapped.nii.gz'  # FIXME not used yet
    output_atlas_annotations_mapped = '/output/atlas_annotations_mapped.nii.gz'  # FIXME not used yet
    output_atlas_template_mapped = '/output/atlas_template_mapped.nii.gz'  # FIXME not used yet
    output_annotated_spec_unmix = '/output/spec_unmix_annotated.csv'

    # voxel_sizes
    voxel_size_sample = [10, 4.7944, 4.7944]
    voxel_size_atlas = 25  # isotropic

    # load raw images
    auto = imread(input_auto_image)  # (z,y,x)
    spec = imread(input_spec_image)  # (z,y,x)

    # unmix signal
    spec_unmix = unmix(auto, spec)

    # downsample autofluorescence to atlas resolution for registration and save for Elastix registration
    auto_25 = downsample_volume(auto, voxel_size_sample, voxel_size_atlas)
    save_nifti(auto_25, output_downsampled_auto)



    # TODO: registration here...
    # TODO: registration here...
    # TODO: registration here...

    # TODO: transform atlas to sample space
    # TODO: transform atlas to sample space
    # TODO: transform atlas to sample space

    # TODO: transform unmixed volume to atlas space
    # TODO: transform unmixed volume to atlas space
    # TODO: transform unmixed volume to atlas space



    # load atlas-to-sample transformed atlas annotations and upsample to full sample resolution
    atlas_annotations_mapped = load_nifti(output_atlas_annotations_mapped)
    atlas_annotations_mapped = upsample_volume(atlas_annotations_mapped, spec_unmix.shape, interp_order=0)

    # get annotation template
    df = pd.read_csv(input_annotation_template, header=None)
    region_ids = df.iloc[:,0].to_numpy()  # get region ID column
    df.iloc[:, 1] = region_wise_quantification(spec_unmix, atlas_annotations_mapped, region_ids)
    df.to_csv(output_annotated_spec_unmix, header=None, index=None)


