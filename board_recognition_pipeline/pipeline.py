from board_recognition import train, eval
import datetime

def main():
    dataset_root = "dataset/images"
    csv_path = "../dataset/corners.csv"

    model_save_path = f"trained_models/board_recognition/keypoint_rcnn_chessboard_{datetime.datetime.now():%Y%m%d_%H%M%S}.pth"

    print("=== START TRAINING ===")
    train.train_model(
        dataset_root=dataset_root,
        csv_path=csv_path,
        num_keypoints=4,
        num_epochs=10,
        batch_size=8,
        num_workers=8,
        lr=0.005,
        save_path=model_save_path
    )

    print("=== START EVALUATION ===")
    eval.evaluate_model(
        model_path=model_save_path,
        dataset_root=dataset_root,
        csv_path=csv_path,
        num_keypoints=4,
        max_images=5
    )

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
