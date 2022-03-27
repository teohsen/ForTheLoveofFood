import subprocess
import argparse

def list_datasets():
    cmd = ["kaggle", "datasets", "list"]

    shell = subprocess.Popen(cmd,
                             shell=True,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True
                             )

    msg = shell.stdout.readlines()
    shell.kill()
    for i in msg:
        print(i)


def download_dataset(dataset_name):
    cmd = ["kaggle", "datasets", "download", dataset_name, "-p", r"data\kaggle_datasets"]

    shell = subprocess.Popen(cmd,
                             shell=True,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True
                             )

    msg = shell.stdout.readlines()
    shell.kill()
    for i in msg:
        print(i)


if __name__ == "__main__":
    # download_dataset("mariotormo/complete-pokemon-dataset-updated-090420")
    # src\utils\kaggle.py -f download_dataset -dn piterfm/2022-ukraine-russian-war

    parser = argparse.ArgumentParser(description='Get some datasets')
    parser.add_argument('-f', metavar='f', type=str)
    parser.add_argument("-dn", default=None, type=str)

    args = parser.parse_args()
    if args.f == "download_dataset":
        assert args.dn is not None
        download_dataset(args.dn)
    elif args.f == "list":
        list_datasets()

