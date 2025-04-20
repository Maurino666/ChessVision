import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from collections import defaultdict

from .dataset import ChessboardCornersDataset
from .train import batch_collate_fn

#This script is used to test the dataset and collate function.

def test_dataset_and_collate(num_workers=2, batches_shown=2, batch_size = 2):
    dataset_root = "dataset/images"
    csv_path = "../dataset/corners.csv"

    dataset = ChessboardCornersDataset(
        root=dataset_root,
        csv_file=csv_path,
        resize_hw=(800, 800)
    )

    print("dataset length:", len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=batch_collate_fn
    )

    # This dictionary maps a 4-tuple (xmin, ymin, xmax, ymax) -> list of occurrences
    # Each occurrence is (batch_idx, target_idx_in_batch, box_idx_in_target)
    box_occurrences = defaultdict(list)

    # Collect data
    total_boxes = 0
    for batch_idx, (images, targets) in enumerate(dataloader):
        for target_idx, t in enumerate(targets):
            boxes_t = t["boxes"]  # shape [M, 4] if you have M boxes
            for box_i, single_box in enumerate(boxes_t):
                box_tuple = tuple(single_box.tolist())  # (xmin, ymin, xmax, ymax)
                box_occurrences[box_tuple].append((batch_idx, target_idx, box_i))
                total_boxes += 1

        if batch_idx >= batches_shown:
            break

    # Summaries:
    zero_box_list = []
    duplicate_list = []

    for box_tuple, occ_list in box_occurrences.items():
        # Check if it's all zeros
        if all(val == 0 for val in box_tuple):
            zero_box_list.append((box_tuple, occ_list))

        # Check if more than one occurrence -> duplicates
        if len(occ_list) > 1:
            duplicate_list.append((box_tuple, occ_list))

    print(f"\n[TEST] num_workers={num_workers}")
    print(f"  Batches viewed (max): {batches_shown + 1}")
    print(f"  Total boxes encountered: {total_boxes}")

    # Report zero boxes
    if zero_box_list:
        print("\n--- ZERO BOXES FOUND ---")
        for (box_tuple, occurrences) in zero_box_list:
            print(f" Box={box_tuple} -> {len(occurrences)} occurrences: {occurrences}")
    else:
        print("\nNo zero boxes found.")

    # Report duplicates
    if duplicate_list:
        print("\n--- DUPLICATE BOXES FOUND ---")
        for (box_tuple, occurrences) in duplicate_list:
            print(f" Box={box_tuple} -> {len(occurrences)} occurrences: {occurrences}")
    else:
        print("\nNo duplicate boxes found.")

def main():
    batches = 20  # how many batches to show at most
    batch_size = 2
    # Single-process first
    print("=== Testing with num_workers=0 (single-process) ===")
    test_dataset_and_collate(num_workers=0, batches_shown=batches, batch_size=batch_size)

    # Multi-process
    print("\n=== Testing with num_workers=2 (multi-process) ===")
    test_dataset_and_collate(num_workers=2, batches_shown=batches, batch_size=batch_size)

    # Multi-process
    print("\n=== Testing with num_workers=4 (even more processes) ===")
    test_dataset_and_collate(num_workers=4, batches_shown=batches, batch_size=batch_size)
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # needed on Windows
    main()
