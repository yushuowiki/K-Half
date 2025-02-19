import csv
from collections import defaultdict
from itertools import combinations
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="GCN for fact classification")
    parser.add_argument('--dataset', type=str, default='GDELT', help='Dataset name')
    return parser.parse_args()


def read_and_group_data(filename):
    data = defaultdict(list)
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if len(row) < 4:
                    continue
                head, relation, tail, timestamp = row[0], row[1], row[2], int(row[3])
                data[(head, relation)].append((tail, timestamp))
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
    return data


def compute_timestamp_differences(group):
    diffs = []
    for (tail1, time1), (tail2, time2) in combinations(group, 2):
        if tail1 != tail2:
            diffs.append(abs(time1 - time2))
    return diffs


def find_mean(differences):
    if not differences:
        return None
    return sum(differences) / len(differences)


def calculate_median(sorted_avg_means):
    median_index = len(sorted_avg_means) // 2
    if len(sorted_avg_means) % 2 == 0:
        return (sorted_avg_means[median_index - 1][1] + sorted_avg_means[median_index][1]) / 2
    else:
        return sorted_avg_means[median_index][1]


def process_data(train_filename):
    grouped_data = read_and_group_data(train_filename)
    result = defaultdict(list)

    for key, group in grouped_data.items():
        diffs = compute_timestamp_differences(group)
        mean_diff = find_mean(diffs)
        if mean_diff is not None:
            result[int(key[1])].append(mean_diff)

    final_result = {}
    for relation, means in result.items():
        if means:
            avg_mean = sum(means) / len(means)
            final_result[relation] = avg_mean

    sorted_avg_means = sorted(final_result.items(), key=lambda x: (x[1] == 0, x[1]))

    non_zero_avg_means = [item for item in sorted_avg_means if item[1] > 0]
    median = calculate_median(non_zero_avg_means)

    print("Average timestamp for each relationship:")
    for relation, avg_mean in sorted_avg_means:
        print(f"The average timestamp for relation {relation}: {avg_mean}")

    print(f"\n Median (excluding relationships with an average timestamp of 0): {median}\n")

    return sorted_avg_means, median


def update_labels(data_filename, short_relations, long_relations):
    data = []
    try:
        with open(data_filename, 'r') as data_file:
            reader = csv.reader(data_file, delimiter='\t')
            for row in reader:
                relation_id = int(row[1])
                if relation_id in short_relations:
                    label = '0'
                elif relation_id in long_relations:
                    label = '1'
                else:
                    label = '1'
                data.append([row[0], relation_id, label])
    except FileNotFoundError:
        print(f"Error: File {data_filename} not found.")
    except Exception as e:
        print(f"Error reading file {data_filename}: {e}")

    try:
        with open(data_filename, 'w', newline='') as data_file:
            writer = csv.writer(data_file, delimiter='\t')
            writer.writerows(data)
    except Exception as e:
        print(f"Error writing file {data_filename}: {e}")

    print(f"{data_filename} Updated label (0 for short term, 1 for long term)）。")


def save_half_life(short_half_life, long_half_life, filename):
    try:
        with open(filename, 'w') as file:
            file.write(f"short_half_life={short_half_life}\n")
            file.write(f"long_half_life={long_half_life}\n")
        print(f"The half-life was saved {filename}")
    except Exception as e:
        print(f"Error saving half-life to {filename}: {e}")

def main():
    args = get_args()

    sorted_avg_means, median = process_data(f'data/{args.dataset}/train.txt')

    short_relations = set()
    long_relations = set()

    short_sum = 0
    long_sum = 0
    short_count = 0
    long_count = 0

    for relation, avg_mean in sorted_avg_means:
        if avg_mean == 0:
            long_relations.add(relation)
        elif avg_mean <= median:
            short_relations.add(relation)
            short_sum += avg_mean
            short_count += 1
        else:
            long_relations.add(relation)
            long_sum += avg_mean
            long_count += 1

    short_half_life = (short_sum / short_count) / 2 if short_count > 0 else 0
    long_half_life = (long_sum / long_count) / 2 if long_count > 0 else 0

    save_half_life(short_half_life, long_half_life, f'data/{args.dataset}/half_life.txt')

    update_labels(f'data/{args.dataset}/relation2id.txt', short_relations, long_relations)


if __name__ == "__main__":
    main()