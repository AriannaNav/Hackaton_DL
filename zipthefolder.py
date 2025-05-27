import argparse
import tarfile
import os

def gzip_folder(folder_path, output_file):
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    print(f" Folder '{folder_path}' has been compressed into '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="Path to folder to compress")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output .tar.gz file")
    args = parser.parse_args()

    gzip_folder(args.folder_path, args.output_file)