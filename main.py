from pathlib import Path
import argparse
from src.computer_vision.cv import main


def define_parser():
    _ = argparse.ArgumentParser(description="Inputs for Object Detection")

    _.add_argument("-fp", help="file path", required=False, default=None, dest="file_path")
    _.add_argument("-b", "--buffer", help="buffer for bounding box", required=False, default=0, type=int, dest="buffer")
    _.add_argument("-o", "--out", required=False, default=False, type=bool, dest="output")
    return _


if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()

    # Some minor checks that the input filepath exists and is an image with ext ".jpg", ".png"
    if args.file_path is not None:
        _ = Path(args.file_path)
        assert _.exists()
        assert _.suffix in [".jpg", ".png"]
        args.file_path = _.as_posix()

    main(**args.__dict__)
