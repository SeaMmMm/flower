import os

# Define the directory path
directory = "/Users/smc/Downloads/flower/datasets/flower/labels/val/"

# Iterate through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)

        # Read the contents of the file
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Modify the lines with the first number as 3
        modified_lines = []
        for line in lines:
            if line.strip().split()[0] == "3":
                modified_lines.append("0" + line.strip()[1:])
            else:
                modified_lines.append(line.strip())

        # Write the modified lines back to the file
        with open(file_path, "w") as file:
            file.write("\n".join(modified_lines))
