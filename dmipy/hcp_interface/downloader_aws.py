import pathlib
import boto
import pkg_resources
import os

HCP_DATA_PATH = pkg_resources.resource_filename(
    'mipy', 'data/hcp/'
)


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
    aws_key : string
        user aws public key
    aws_secret : string
        user aws secret key
    """

    def __init__(self, your_aws_public_key, your_aws_secret_key):
        self.public_key = your_aws_public_key
        self.secret_key = your_aws_secret_key

        s3 = boto.connect_s3(
            aws_access_key_id=self.public_key,
            aws_secret_access_key=self.secret_key,
        )
        self.s3_bucket = s3.get_all_buckets()[1]

    def download_and_prepare_dmipy_example_dataset(self):
        subject = 100307

        directory = os.path.join(HCP_DATA_PATH, str(subject))

        if not os.path.exists(directory):
            os.makedirs(directory)

        for key in self.s3_bucket.list("HCP_1200"):
            path = pathlib.Path(key.name)
            if (
                len(path.parts) == 5 and
                subject == int(path.parts[1]) and
                path.parts[-2] == "Diffusion"
            ):
                if (
                    'bval' in path.parts[-1] or
                    'bvec' in path.parts[-1] or
                    'data' in path.parts[-1] or
                    'nodif' in path.parts[-1]
                ):
                    filepath = os.path.join(directory, path.parts[-1])
                    with open(filepath, 'wb') as f:
                        key.get_contents_to_file(f)

    def download_hcp_dataset(self, subject):
        """
        Parameters
        ----------
        subject: integer
            the identification number of the HCP subject to download
        """


def download(aws_key, aws_secret, subject=100307):

    for key in s3_bucket.list("HCP_1200"):
        path = pathlib.Path(key.name)
        if (
            len(path.parts) == 5 and
            subject == int(path.parts[1]) and
            path.parts[-2] == "Diffusion"
        ):
            if (
                'bval' in path.parts[-1] or
                'bvec' in path.parts[-1] or
                'data' in path.parts[-1] or
                'nodif' in path.parts[-1]
            ):

                with open(path.parts[-1], 'wb') as f:
                    key.get_contents_to_file(f)
