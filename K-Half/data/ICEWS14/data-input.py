import os


def process_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        cols = line.strip().split('\t')
        if len(cols) < 5:
            continue

        cols[3] = str(int(float(cols[3])))
        processed_line = '\t'.join(cols[:4])
        processed_lines.append(processed_line)

    with open(file_name, 'w') as f:
        for line in processed_lines:
            f.write(line + '\n')


def main():
    files_to_process = ['train.txt', 'valid.txt', 'test.txt']

    for file_name in files_to_process:
        if os.path.exists(file_name):
            print(f"Processing {file_name}...")
            process_file(file_name)
        else:
            print(f"{file_name} does not exist in the current directory.")


if __name__ == "__main__":
    main()