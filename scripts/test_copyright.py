# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import os

COPYRIGHT_TAG = "Copyright © 2021 United States Government as represented by the Administrator"  # String to check file lines for


def check_copyright(directory: str, invalid_files: list) -> bool:
    result = True

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)

        # If path is subdirectory, recursively check files/subdirectories within
        if os.path.isdir(path):
            result = result and check_copyright(path, invalid_files)
        # If path is a file, ensure it is of type py and check for copyright
        elif os.path.isfile(path) and path.split(".")[-1] in ("py", "pyw", "py3"):
            file = open(path, "r")
            copyright_met = False
            # Iterate over lines in file, check each line against COPYRIGHT_TAG
            for line in file:
                if COPYRIGHT_TAG in line:
                    # File contains copyright, skip rest of lines
                    file.close()
                    copyright_met = True
                if copyright_met:
                    break
            if not copyright_met:
                result = False
                invalid_files.append(path)
            file.close()

    return result


def main():
    print("\n\nTesting Files for Copyright Information")

    root = "../progpy"
    invalid_files = []
    copyright_confirmed = check_copyright(root, invalid_files)

    if not copyright_confirmed:
        raise Exception(
            f"Failed test\nFiles missing copyright information: {invalid_files}"
        )
    else:
        print("No missing copyright statements")


if __name__ == "__main__":
    main()
