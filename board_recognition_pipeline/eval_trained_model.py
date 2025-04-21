from board_recognition.eval import evaluate_model

def main():
    """
    Esegue solo la fase di evaluation su un modello già addestrato,
    senza eseguire alcun training.
    """
    # Parametri di base per l'evaluation
    model_path = "trained_models/board_recognition/keypoint_rcnn_chessboard_20250420_182043.pth"  # Percorso al modello salvato
    dataset_root = "datasets/images"              # Cartella con le immagini
    csv_path = "datasets/corners.csv"  # File CSV
    num_keypoints = 4                            # Numero di keypoint
    device = None                                 # 'cuda' o 'cpu' (se None, autodetect)
    max_images = 100                           # Quante immagini mostrare

    print("=== START EVALUATION ONLY ===")
    evaluate_model(
        model_path=model_path,
        dataset_root=dataset_root,
        csv_path=csv_path,
        num_keypoints=num_keypoints,
        device=device,
        max_images=max_images
    )
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
