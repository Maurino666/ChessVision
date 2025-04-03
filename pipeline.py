import train
import eval
import datetime

def main():
    # Parametri di base (puoi modificarli a piacere)
    dataset_root = "dataset/images"
    csv_path = "dataset/corners.csv"

    # Genera un nome univoco per il modello salvato
    datetime_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"trained_models/keypoint_rcnn_chessboard_{datetime_suffix}.pth"

    num_keypoints = 4

    # 1) Fase di training
    print("=== START TRAINING ===")
    train.train_model(
        dataset_root=dataset_root,
        csv_path=csv_path,
        num_keypoints=num_keypoints,
        num_epochs=10,
        batch_size=2,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
        device=None,  # auto-detect CUDA
        save_path=model_save_path
    )

    # 2) Fase di evaluation
    print("=== START EVALUATION ===")
    eval.evaluate_model(
        model_path=model_save_path,
        dataset_root=dataset_root,
        csv_path=csv_path,
        num_keypoints=num_keypoints,
        device=None,  # auto-detect
        max_images=5
    )

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
