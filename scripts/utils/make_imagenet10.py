import os, sys

# usage: python make_imagenet100.py full/imagenet/path desired/imagenet100/path
full_imagenet_path = sys.argv[1]
imagenet100_path = sys.argv[2]

class_names = [
    "n01440764",  # tench
    "n02102040",  # English springer
    "n02979186",  # cassette player
    "n03000684",  # chain saw
    "n03028079",  # church
    "n03394916",  # French horn
    "n03417042",  # garbage truck
    "n03425413",  # gas pump
    "n03445777",  # golf ball
    "n03888257",  # parachute
]


for subdir in ["train", "val"]:
    os.makedirs(os.path.join(imagenet100_path, subdir), exist_ok=True)
    for class_name in class_names:
        os.symlink(
            os.path.join(full_imagenet_path, subdir, class_name),
            os.path.join(imagenet100_path, subdir, class_name),
            target_is_directory=True,
        )
