import sys
import subprocess
from pathlib import Path

def print_context(filenames, i):
    # Print the lines around the current index
    print("Context:")
    for j in range(i-2, i+3):
        if j < 0 or j >= len(filenames):
            continue
        print(j, filenames[j])


def check_filename_list(filenames):
    all_good = True
    stems_set = set()

    # Iterate every two lines
    for i in range(0, len(filenames), 2):
        line1 = filenames[i]
        line2 = filenames[i+1]

        # Convert both to path
        path1 = Path(line1)
        path2 = Path(line2)

        # Check that stems match
        if path1.stem != path2.stem:
            print("Stems do not match: ", path1, path2)
            print_context(filenames, i)
            all_good = False

        # Check that stem didn't show up already
        if path1.stem in stems_set:
            print("Stem already seen: ", path1)
            print_context(filenames, i)
            all_good = False
        else:
            stems_set.add(path1.stem)

    return all_good


def main():
    # Get listing from tar file
    filename = sys.argv[1]

    result = subprocess.run(["tar", "-tf", filename], check=True, capture_output=True)
    lines = result.stdout.splitlines()
    filenames = [l.decode('utf-8') for l in lines]
    if check_filename_list(filenames):
        print(f"{filename}: All OK")


if __name__ == "__main__":
    main()