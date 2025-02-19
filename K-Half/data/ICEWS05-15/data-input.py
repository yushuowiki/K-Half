import os

def process_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        cols = line.strip().split('\t')
        if len(cols) >= 4:
            # 将第四列内容变为整数
            cols[3] = str(int(float(cols[3])))
            # 只保留前四列
            processed_line = '\t'.join(cols[:4])
            processed_lines.append(processed_line)

    # 将处理后的内容写回文件
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