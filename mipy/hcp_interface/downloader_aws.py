import pathlib
import boto


def download(aws_key, aws_secret, subject=100307):
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
    subject : integer
        the identification number of the HCP subject to download

    """
    s3 = boto.connect_s3(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
    )

    s3_bucket = s3.get_all_buckets()[1]
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
