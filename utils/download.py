"""A script to make downloading the DiffusionDB dataset easier."""
from urllib.error import HTTPError
from urllib.request import urlretrieve
from os.path import exists
import sys
import shutil
import os
import time
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob
import zipfile



index = None  # initiate main arguments as None
range_max = None
output = None
unzip = None
large = None

parser = argparse.ArgumentParser(description="Download a file from a URL")  #

# It's adding arguments to the parser.
parser.add_argument(
    "-i",
    "--index",
    type=int,
    default=1,
    help="File to download or lower bound of range if -r is set",
)
parser.add_argument(
    "-r",
    "--range",
    type=int,
    default=14000,
    help="Upper bound of range if -i is provided",
)
parser.add_argument(
    "-o", "--output", type=str, default="dataset/StableDiffusionDB", help="Output directory name"
)
parser.add_argument(
    "-z",
    "--unzip",
    help="Unzip the file after downloading",
    # It's setting the argument to True if it's provided.
    action="store_true",
)
parser.add_argument(
    "-d",
    "--download",
    help="Download from DiffusionDB Large (14 million images)",
    action="store_true",
)
parser.add_argument(
    "-n",
    "--n-jobs",
    type=int,
    default=4,
    help="n_jobs"
)

args = parser.parse_args()  # parse the arguments

# It's checking if the user has provided any arguments, and if they have, it
# sets the variables to the arguments.
if args.index:
    index = args.index
if args.range:
    range_max = args.range
if args.output:
    output = args.output
if args.n_jobs:
    n_jobs = args.n_jobs

unzip = args.unzip
download = args.download

def callbackfunc(blocknum, blocksize, totalsize):
    '''回调函数
    @blocknum: 已经下载的数据块
    @blocksize: 数据块的大小
    @totalsize: 远程文件的大小
    '''
    sys.stdout.write('\r>> Downloading %.1f%%' % (float(blocknum * blocksize) / float(totalsize) * 100.0))
    sys.stdout.flush()


def download_from_url(url, output=output):

    if output != "":
        output = f"{output}/"

    if not exists(output):
        os.makedirs(output)

    name = url.split("/")[-1]
    file_path = f"{output}{name}"
    if os.path.exists(file_path):
        flag = check_zipfile(file_path)
        if flag:
            print(f"{file_path} has exists, pass it")
            return 
        else:
            print(f"{file_path} is wrong, redownload")
        

    print("Downloading file: ", url)
    try:
        urlretrieve(url, file_path)
        flag = check_zipfile(file_path)
        while not flag:
            print(f"{file_path} error ,redownload...")
            urlretrieve(url, file_path)
            flag = check_zipfile(file_path)
        print(f"{file_path} download successfully!")
        return
    except Exception as e:
        print(f"Encountered an Error downloading file: {url} - {e}")
        return url

def check_zipfile(filename):
    try:
        with zipfile.ZipFile(filename) as zf:
            return zf.testzip() is None
    except Exception as e:
        
        return False
    

def unzip_file(file: str):
    """
    > This function takes a zip file as an argument and unpacks it
    :param file: str
    :type file: str
    :return: The file name without the .zip extension
    """
    if exists(file[:-4]):
        if len(glob(file[:-4]+"/*")) != 1001:
            print(f'{file[:-4]} may be miss some?')
            shutil.rmtree(file[:-4])
             
    if not os.path.exists(file[:-4]):
        os.makedirs(file[:-4])
    

    try:
        shutil.unpack_archive(file, file[:-4])
        # print(file[:-4]+"/*")
        if len(glob(file[:-4]+"/*")) == 1001:
            print(f"File: {file.replace('.zip', '')} has been unzipped")
            os.remove(file)
            return 
        else:
            print(f"File: {file.replace('.zip', '')} maybe has error")
            return file
    except Exception as e:
        print(e)
        return file



if __name__ == "__main__":

    if range_max - index >= 13999:
        confirmation = print("make sure you have at least 6.7Tb free")


    baseurl = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/"
    download_url = []
    if download:

        for idx in range(index, range_max+1):
        #     if i <= 10000:
        #         url = f"{baseurl}diffusiondb-large-part-1/part-{i:06}.zip"
        #     else:
        #         url = f"{baseurl}diffusiondb-large-part-2/part-{i:06}.zip"
            url = f"{baseurl}images/part-{idx:06}.zip"
            download_url.append(url)
        
        # if exists('error_url.txt'):
        #     with open('error_url.txt', 'r') as f:
        #         download_url = f.readlines()
        print(len(download_url))
        fail_url = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(download_from_url)(url) for url in tqdm(download_url, total=len(download_url)))
        fail_url = [x for x in fail_url if x is not None]
        with open("error_url.txt", 'w') as f:
            f.write('\n'.join(fail_url))

    if unzip:
        unzip_path = glob(output+"/*.zip")
       
        file_list = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(unzip_file)(path) for path in tqdm(unzip_path, total=len(unzip_path)))
        file_list = [x for x in file_list if x is not None]
        with open("error_file.txt", 'w') as f:
            f.write('\n'.join(file_list))