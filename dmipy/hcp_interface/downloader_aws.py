import pathlib
import boto
import pkg_resources
import os
import nibabel as nib
import numpy as np


DATA_PATH = pkg_resources.resource_filename(
    'dmipy', 'data'
)

__all__ = [
    'HCPInterface'
]


class HCPInterface:
    """
    Function to download HCP data directly from AWS server.
    Since HCP requires you to sign its user agreements, this function
    takes as input the user's personal public and secret AWS key.

    These keys can be generated following the the instructions here:
    https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS

    The reason this function is set up this way is because we cannot sign
    the user agreement for the user. Now, the user takes its own responsibility
    and we can provide this convenient downloader without having to go through
    the HCP interface.

    Parameters
    ----------
    your_aws_public_key : string
        user aws public key
    your_aws_secret_key : string
        user aws secret key
    """

    def __init__(self, your_aws_public_key, your_aws_secret_key):
        self.public_key = your_aws_public_key
        self.secret_key = your_aws_secret_key

        s3 = boto.connect_s3(
            aws_access_key_id=self.public_key,
            aws_secret_access_key=self.secret_key,
        )

        for key in s3.get_all_buckets():
            if key.name == 'hcp-openaccess':
                self.s3_bucket = key
                break

        self.hcp_directory = os.path.join(DATA_PATH, 'hcp')
        if not os.path.exists(self.hcp_directory):
            os.makedirs(self.hcp_directory)

    def download_and_prepare_dmipy_example_dataset(self):
        """
        Downloads subect 100307 of the Wu-Minn HCP data and prepares it to be
        used for the dmipy example notebooks.
        """
        subject_ID = 100307
        self.download_subject(subject_ID)
        self.prepare_example_slice(subject_ID)

    @property
    def available_subjects(self):
        subjects_textfile = os.path.join(DATA_PATH, "wu_minn_hcp_subjects.txt")
        return np.loadtxt(subjects_textfile, dtype=int)

    def download_subject(self, subject_ID):
        """
        Downloads Wu-Minn HCP subject data to the dmipy data folder.
        The downloaded data includes the b-values, gradient orientations,
        diffusion-weighted images and the binary brain mask.

        Parameters
        ----------
        subject_ID: integer
            the identification number of the Wu-Minn HCP subject
        """
        hcp_data_path = os.path.join(self.hcp_directory, str(subject_ID))

        if not os.path.exists(hcp_data_path):
            os.makedirs(hcp_data_path)

        print('Downloading data to {}'.format(hcp_data_path))

        counter = 0
        for key in self.s3_bucket.list("HCP_1200"):
            path = pathlib.Path(key.name)
            if (
                len(path.parts) == 5 and
                subject_ID == int(path.parts[1]) and
                path.parts[-2] == "Diffusion"
            ):
                if (
                    'bval' in path.parts[-1] or
                    'bvec' in path.parts[-1] or
                    'data' in path.parts[-1] or
                    'nodif' in path.parts[-1]
                ):
                    print('Downloading {}'.format(path.parts[-1]))
                    filepath = os.path.join(hcp_data_path, path.parts[-1])
                    with open(filepath, 'wb') as f:
                        key.get_contents_to_file(f)
                    counter += 1
                    if counter == 4:
                        break

    def prepare_example_slice(self, subject_ID):
        "Prepares a coronal slice for the dmipy example notebooks."
        msg = "Preparing coronal slice for dmipy examples"
        print(msg)

        folder_name = "hcp_example_slice"
        example_directory = os.path.join(self.hcp_directory, folder_name)
        if not os.path.exists(example_directory):
            os.makedirs(example_directory)

        subject_data_path = os.path.join(self.hcp_directory, str(subject_ID))

        data = nib.load(os.path.join(
            subject_data_path, 'data.nii.gz')).get_data()
        affine = nib.load(os.path.join(
            subject_data_path, 'data.nii.gz')).affine
        mask = nib.load(os.path.join(
            subject_data_path, 'nodif_brain_mask.nii.gz')).get_data()
        data_shape = data.shape

        slice_index = data_shape[1] // 2
        data_slice = data[:, slice_index: slice_index + 1]
        mask_slice = mask[:, slice_index: slice_index + 1]
        data_slice[mask_slice == 0] = 0

        nib.save(nib.Nifti1Image(data_slice, affine), os.path.join(
            example_directory, 'coronal_slice.nii.gz'))
        print('Done')
