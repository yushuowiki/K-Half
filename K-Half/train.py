import torch
import dgl
from GCN import GCN
from K_Half import K_Half
from sklearn.model_selection import train_test_split
import argparse
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="GCN for fact classification")
    parser.add_argument('--dataset', type=str, default='GDELT', help='Dataset name')
    parser.add_argument('--train_file', type=str, default='train', help='train、test、valid')
    parser.add_argument('--initial_entity_emb_dim', type=int, default=128, help='Dimension of initial entity embeddings')
    parser.add_argument('--entity_out_dim_1', type=int, default=128, help='Entity output embedding dimension (layer 1)')
    parser.add_argument('--entity_out_dim_2', type=int, default=128, help='Entity output embedding dimension (layer 2)')
    parser.add_argument('--h_dim', type=int, default=32, help='Dimension of time embeddings')
    parser.add_argument('--num_ents', type=int, default=8000, help='Number of entities')
    parser.add_argument('--nheads_GAT_1', type=int, default=4, help='Number of GAT heads (layer 1)')
    parser.add_argument('--nheads_GAT_2', type=int, default=4, help='Number of GAT heads (layer 2)')
    parser.add_argument('--n_hidden', type=int, default=128, help='Hidden layer dimension for GCN')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for GCN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--batch', type=int, default=10000, help='batch_size')
    parser.add_argument('--threshold', type=int, default=250, help='Threshold for minimum occurrences')
    return parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_second_column_occurrences(file_path):
    with open(file_path, 'r') as file:
        second_column_data = [line.split()[1] for line in file]
    return Counter(second_column_data)

def remove_less_than_threshold(occurrences, threshold):
    return {key: value for key, value in occurrences.items() if value >= threshold}

def print_occurrences(occurrences):
    sorted_occurrences = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)
    total_occurrences = sum(occurrences.values())

    for key, value in sorted_occurrences:
        print(f"serial number: {key}, Number of occurrences: {value}")

    print(f"\n total: {total_occurrences}")

def overwrite_file_with_filtered(file_path, filtered_occurrences):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            second_column_value = line.split()[1]
            if second_column_value in filtered_occurrences:
                file.write(line)


def load_relation_data(file, train_file, emb_dim):
    valid_relation_ids = set()

    with open(train_file, 'r') as f:
        for line in f:
            relation_id = int(line.split('\t')[1])
            valid_relation_ids.add(relation_id)

    relation_dict = {}
    labels = {}

    with open(file, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            relation_semantic, relation_id, label = cols[0], int(cols[1]), int(cols[2])

            if relation_id in valid_relation_ids:
                relation_emb = torch.randn(emb_dim)

                relation_dict[relation_id] = relation_emb
                labels[relation_id] = label

    return relation_dict, labels

def load_train_data(train_file, entity_id_map):
    facts = []
    with open(train_file, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            head_entity, relation_id, tail_entity, timestamp = int(cols[0]), int(cols[1]), int(cols[2]), float(cols[3])

            if head_entity in entity_id_map and tail_entity in entity_id_map:
                mapped_head = entity_id_map[head_entity]
                mapped_tail = entity_id_map[tail_entity]
                facts.append((mapped_head, relation_id, mapped_tail, timestamp))
    return facts

def create_entity_id_map(train_file):
    entity_id_map = {}
    current_index = 0
    with open(train_file, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            head_entity = int(cols[0])
            tail_entity = int(cols[2])

            if head_entity not in entity_id_map:
                entity_id_map[head_entity] = current_index
                current_index += 1
            if tail_entity not in entity_id_map:
                entity_id_map[tail_entity] = current_index
                current_index += 1

    return entity_id_map


def construct_graph(facts):
    g = dgl.graph(([], []))
    g.add_nodes(len(facts))

    edges = []
    for i, (_, rel1, _, _) in enumerate(facts):
        for j, (_, rel2, _, _) in enumerate(facts):
            if i != j and rel1 == rel2:
                edges.append((i, j))

    if edges:
        src, dst = zip(*edges)
        g.add_edges(src, dst)

    return g


def generate_fact_embeddings(facts, entity_emb, relation_dict, his_temp_embs=None):
    fact_embeddings = []
    for i, (head, relation_id, tail, _) in enumerate(facts):
        entity_emb_head = entity_emb[head]
        entity_emb_tail = entity_emb[tail]
        relation_emb = relation_dict[relation_id]
        time_emb = his_temp_embs[i][head]

        fact_emb = torch.cat([entity_emb_head, entity_emb_tail, relation_emb, time_emb], dim=0)

        fact_embeddings.append(fact_emb)

    return torch.stack(fact_embeddings)

def compute_accuracy(logits, labels):
    _, predicted = torch.max(logits, dim=1)

    correct = (predicted == labels).sum().item()

    total = labels.size(0)

    return correct / total


def write_predictions_to_file(facts, predictions, dataset, train_file, reverse_entity_id_map):

    file_path = f"data/{dataset}/{train_file}.txt"

    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path, 'w') as f:
        for i, line in enumerate(lines):
            line = line.strip()
            mapped_head, relation_id, mapped_tail, timestamp = facts[i]

            original_head = reverse_entity_id_map[mapped_head]
            original_tail = reverse_entity_id_map[mapped_tail]

            prediction = predictions[i]
            f.write(f"{original_head}\t{relation_id}\t{original_tail}\t{timestamp}\t{prediction}\n")

    print(f"Predictions appended to {file_path}")

def create_reverse_entity_id_map(entity_id_map):
    return {v: k for k, v in entity_id_map.items()}


def main(args):
    file_path = f'data/{args.dataset}/{args.train_file}.txt'
    occurrences = count_second_column_occurrences(file_path)
    filtered_occurrences = remove_less_than_threshold(occurrences, args.threshold)
    print_occurrences(filtered_occurrences)
    overwrite_file_with_filtered(file_path, filtered_occurrences)
    print(f"File filtered and overwritten: {file_path}")

    entity_id_map = create_entity_id_map(file_path)
    reverse_entity_id_map = create_reverse_entity_id_map(entity_id_map)

    relation_dict, relation_labels = load_relation_data(f'data/{args.dataset}/relation2id.txt', file_path, args.entity_out_dim_1)
    facts = load_train_data(file_path, entity_id_map)

    initial_entity_emb = torch.randn(args.num_ents, args.initial_entity_emb_dim)
    entity_out_dim = [args.entity_out_dim_1, args.entity_out_dim_2]
    nheads_GAT = [args.nheads_GAT_1, args.nheads_GAT_2]
    k_half_model = K_Half(initial_entity_emb, entity_out_dim, args.h_dim, args.num_ents, nheads_GAT,
                          relation_dict=relation_dict)

    edge_list = torch.tensor([[fact[0], fact[2]] for fact in facts]).t()
    edge_type = torch.tensor([fact[1] for fact in facts])

    all_predictions = []
    batch_size = args.batch
    num_edges = edge_list.shape[1]

    train_losses = []
    val_losses = []
    val_accuracies = []

    for start in range(0, num_edges, batch_size):
        end = min(start + batch_size, num_edges)
        edge_list_batch = edge_list[:, start:end]
        edge_type_batch = edge_type[start:end]
        batch_inputs = torch.tensor([[fact[0], fact[1], fact[2], fact[3]] for fact in facts[start:end]])

        entity_emb, his_temp_embs = k_half_model(Corpus_=None, batch_inputs=batch_inputs, edge_list=edge_list_batch,
                                                 edge_type=edge_type_batch)

        fact_embeddings = generate_fact_embeddings(facts[start:end], entity_emb, relation_dict, his_temp_embs)

        labels = torch.tensor([relation_labels[fact[1]] for fact in facts[start:end]])

        train_idx, test_idx = train_test_split(list(range(len(facts[start:end]))), test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)

        g = construct_graph(facts[start:end])

        gcn_model = GCN(g, in_feats=fact_embeddings.shape[1], n_hidden=args.n_hidden, n_classes=args.n_classes,
                        n_layers=args.n_layers, activation=F.relu, dropout=args.dropout)

        optimizer = torch.optim.Adam(gcn_model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            gcn_model.train()
            logits = gcn_model(fact_embeddings)
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                gcn_model.eval()
                val_logits = gcn_model(fact_embeddings)
                val_loss = F.cross_entropy(val_logits[val_idx], labels[val_idx])
                val_accuracy = compute_accuracy(val_logits[val_idx], labels[val_idx])

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            val_accuracies.append(val_accuracy)

            print(
                f'Epoch {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}, Val Accuracy: {val_accuracy * 100:.2f}%')

            if val_accuracy == 1.0:
                print("Validation accuracy reached 100%, stopping training early.")
                break

            gcn_model.eval()

        with torch.no_grad():
            predictions = torch.argmax(gcn_model(fact_embeddings), dim=1)
            all_predictions.extend(predictions.tolist())

    write_predictions_to_file(facts, all_predictions, args.dataset, args.train_file, reverse_entity_id_map)



if __name__ == "__main__":
    args = get_args()
    main(args)
