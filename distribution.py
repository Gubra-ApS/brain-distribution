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


class Elastix:
    def __init__(self, moving, fixed, elastix_path, result_path):
        self.moving = moving
        self.fixed = fixed
        self.elastix_path = elastix_path
        self.result_path = result_path

    def registration(self, params, result_name, init_trans=r'', f_mask=r'', save_nifti=True, datatype='uint16', origin_vol=False):
        program = r'elastix -threads 16 '; #elastix is added to PATH
        fixed_name = r'-f ' + self.fixed + r' ';
        moving_name = r'-m ' + self.moving + r' ';
        outdir = r'-out ' + self.elastix_path + r'/workingDir ';
        params = r'-p ' + self.elastix_path + r'/' + params + r' ';

        try:
            if f_mask!=r'':
                fmask =  r'-fMask ' + f_mask
                if init_trans != r'':
                    t0 = r'-t0 ' + os.path.join(self.result_path,init_trans + r'.txt ')
                    os.system(program + fixed_name + moving_name  + outdir + params + t0 + fmask)
                else:
                    os.system(program + fixed_name + moving_name  + outdir + params + fmask)
                    print(program + fixed_name + moving_name  + outdir + params + fmask)
            else:
                if init_trans != r'':
                    t0 = r'-t0 ' + os.path.join(self.result_path,init_trans + r'.txt ')
                    os.system(program + fixed_name + moving_name  + outdir + params + t0)
                else:
                    print(program + fixed_name + moving_name  + outdir + params)
                    os.system(program + fixed_name + moving_name  + outdir + params)

            move(self.elastix_path + r'/workingDir/TransformParameters.0.txt', os.path.join(self.result_path,result_name + r'.txt'))

            if save_nifti==True:
                move(self.elastix_path + r'/workingDir/result.0.nii.gz', os.path.join(self.result_path,result_name + r'.nii.gz'))

                ## temp hack as the nifti file from elastix could not be opened in ITK-SNAP
                temp = sitk.ReadImage(os.path.join(self.result_path,result_name + r'.nii.gz'))
                temp = sitk.GetArrayFromImage(temp)
                temp[temp<0] = 0
                if datatype=='uint16':
                    temp[temp>65535] = 65535
                    temp = temp.astype('uint16')
                elif datatype=='uint8':
                    temp[temp>255] = 255
                    temp = temp.astype('uint8')

                final_sitk = sitk.GetImageFromArray(temp)
                if isinstance(origin_vol, bool)==False:
                    sitk_origin =sitk.GetImageFromArray(origin_vol)
                    final_sitk.SetOrigin((round(sitk_origin.GetWidth()/2), round(sitk_origin.GetHeight()/2), -round(sitk_origin.GetDepth()/2)))
                else:
                    final_sitk.SetOrigin((round(final_sitk.GetWidth()/2), round(final_sitk.GetHeight()/2), -round(final_sitk.GetDepth()/2)))

                sitk.WriteImage(final_sitk,os.path.join(self.result_path,result_name + r'.nii.gz'))

        finally:
            # optional clean up code
            pass

        return True


'''
### Run unmixing example script
'''
if __name__ == '__main__':

    # elastix folder
    folder_output_elastix = '/elastix_folder' # folder with registration param files etc.

    # input files
    input_auto_image = '/input/auto.tif'  # autofluorescence
    input_spec_image = '/input/spec.tif'  # specific channel
    input_atlas_template = '/input/atlas_template.nii.gz'
    input_atlas_annotations = '/input/atlas_annotations.nii.gz'
    input_annotation_template = '/input/annotation_template.csv'

    # output files
    output_downsampled_auto = '/output/auto_downsampled.nii.gz'
    output_spec_unmix = '/output/spec_unmix.nii.gz'
    output_spec_unmix_mapped = '/output/spec_unmix_mapped.nii.gz'
    output_atlas_annotations_mapped = '/output/atlas_annotations_mapped.nii.gz'
    output_atlas_template_mapped = '/output/atlas_template_mapped.nii.gz'
    output_annotated_spec_unmix = '/output/spec_unmix_annotated.csv'

    # voxel_sizes
    voxel_size_sample = [10, 4.7944, 4.7944]
    voxel_size_atlas = 25  # isotropic

    # load raw images
    auto = imread(input_auto_image)  # (z,y,x)
    spec = imread(input_spec_image)  # (z,y,x)

    # unmix signal
    spec_unmix = unmix(auto, spec)
    save_nifti(spec_unmix, output_spec_unmix)

    # downsample autofluorescence to atlas resolution for registration and save for Elastix registration
    auto_25 = downsample_volume(auto, voxel_size_sample, voxel_size_atlas)
    save_nifti(auto_25, output_downsampled_auto)

    ## Atlas registration
    # Parameter files
    params_affine = 'affine_mouse_atlas_v6.1.txt'
    params_bspline = 'bspline_mouse_atlas_v6.1.txt'

    # DEFINE OUTPUT FILES: transformation files to be created
    file_a2s_affine = '_regi_atlas_2_sample_affine'
    file_a2s_bspline = output_atlas_template_mapped

    file_s2a_affine = '_regi_sample_2_atlas_affine'
    file_s2a_bspline = output_spec_unmix_mapped

    # Align atlas to sample space
    file_moving = input_atlas_template
    file_fixed = output_downsampled_auto
    hu_a2s = Elastix(file_moving, file_fixed, folder_output_elastix, 'output')

    hu_a2s.registration(params=params_affine, result_name=file_a2s_affine, datatype='uint8')
    hu_a2s.registration(params=params_bspline, result_name=file_a2s_bspline, init_trans=file_a2s_affine,
                        datatype='uint8')

    # Transform atlas annotations to sample space
    path_input = input_atlas_annotations
    path_output = output_atlas_annotations_mapped
    hu_a2s.transform_vol(volume=path_input, trans_params=file_a2s_bspline, result_name=path_output, type='ano')

    # Align sample to atlas space
    file_moving = output_downsampled_auto
    file_fixed = input_atlas_template
    hu_s2a = Elastix(file_moving, file_fixed, folder_output_elastix, 'output')

    hu_s2a.registration(params=params_affine, result_name=file_s2a_affine, datatype='uint8')
    hu_s2a.registration(params=params_bspline, result_name=file_s2a_bspline, init_trans=file_s2a_affine,
                        datatype='uint8')

    # Transform unmix signal to atlas space
    path_input = output_spec_unmix
    path_output = output_spec_unmix_mapped
    hu_s2a.transform_vol(volume=path_input, trans_params=file_s2a_bspline, result_name=path_output, type='vol')


    ## Quantification
    # load atlas-to-sample transformed atlas annotations and upsample to full sample resolution
    atlas_annotations_mapped = load_nifti(output_atlas_annotations_mapped)
    atlas_annotations_mapped = upsample_volume(atlas_annotations_mapped, spec_unmix.shape, interp_order=0)

    # get annotation template for quantification
    df = pd.read_csv(input_annotation_template, header=None)
    region_ids = df.iloc[:,0].to_numpy()  # get region ID column
    df.iloc[:, 1] = region_wise_quantification(spec_unmix, atlas_annotations_mapped, region_ids)
    df.to_csv(output_annotated_spec_unmix, header=None, index=None)
