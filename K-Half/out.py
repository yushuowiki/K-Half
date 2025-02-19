import math
import argparse
from tqdm import tqdm


def load_train_data(train_file):
    facts = []
    with open(train_file, 'r') as f:
        for line in tqdm(f, desc="Loading data", unit="line"):
            cols = line.strip().split('\t')

            head_entity = int(cols[0])
            relation_id = int(cols[1])
            tail_entity = int(cols[2])
            timestamp = float(cols[3])
            label = int(cols[4])

            facts.append((head_entity, relation_id, tail_entity, timestamp, label))

    return facts


def save_facts(file_path, facts):
    with open(file_path, 'w') as f:
        for fact in tqdm(facts, desc="Saving facts", unit="fact"):
            f.write('\t'.join(map(str, fact)) + '\n')
        f.flush()


def save_all_output(file_path, output_data):
    with open(file_path, 'w') as f:
        f.write(output_data)
    print(f"All output saved to {file_path}")


def calculate_effectiveness(V0, t, half_life):
    decay_constant = math.log(2) / half_life
    return V0 * math.exp(-decay_constant * abs(t))



def filter_facts(facts, threshold, short_half_life, long_half_life):
    filtered_facts = set()
    facts_dict = {}
    output_data = ""

    for fact in tqdm(facts, desc="Filtering facts", unit="fact"):
        key = (fact[0], fact[1])
        if key not in facts_dict:
            facts_dict[key] = []
        facts_dict[key].append(fact)

    removed_facts = []

    for key, fact_group in facts_dict.items():
        output_data += f"\nProcessing group: Head entity {key[0]}, Relation {key[1]}\n"
        fact_group.sort(key=lambda x: x[3], reverse=True)
        latest_fact = fact_group[0]
        t0 = latest_fact[3]
        V0 = 1

        output_data += f"Latest fact: {latest_fact}, using this fact as reference for comparison.\n"

        for fact in fact_group:
            if fact[:3] == latest_fact[:3]:
                output_data += f"Fact {fact} matches the latest fact (same head, relation, tail), keeping it.\n"
                continue

            ti = fact[3]
            t = ti - t0

            if fact[4] == 0:
                half_life = short_half_life
            else:
                half_life = long_half_life

            V = calculate_effectiveness(V0, t, half_life)
            output_data += f"Fact: {fact}, Time difference (t): {t}, Effectiveness: {V}\n"

            if V > threshold:
                V0 = 1
            else:
                filtered_facts.add(fact)
                removed_facts.append(fact)
                output_data += f"Filtered out fact: {fact}\n"

    return [fact for fact in facts if fact not in filtered_facts], removed_facts, output_data


def load_half_life(filename):
    half_life = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                half_life[key] = float(value)
        print(f" {filename} half-life : {half_life}")
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"Error reading half-life from {filename}: {e}")

    return half_life.get('short_half_life', 0), half_life.get('long_half_life', 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True, help="validity threshold")
    args = parser.parse_args()

    file_path = f'data/{args.dataset}/{args.train_file}.txt'
    short_half_life, long_half_life = load_half_life(f'data/{args.dataset}/half_life.txt')
    facts = load_train_data(file_path)


    filtered_facts, removed_facts, output_data = filter_facts(facts, args.threshold, short_half_life, long_half_life)

    save_facts(file_path, filtered_facts)

    output_file_path = f'data/{args.dataset}/{args.train_file}_output.txt'
    save_all_output(output_file_path, output_data)


if __name__ == "__main__":
    main()