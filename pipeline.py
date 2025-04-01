import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# Importa il dataset e i trasformatori definiti nei rispettivi file
from dataset import ChessboardCornersDataset
from transforms import (
    get_train_augmentations,
    KorniaTransformWrapper,
    get_test_augmentations  # Definiremo questa funzione nel file transforms.py
)


# Funzione di collate per raggruppare immagini e target in liste
def collate_fn(batch):
    return list(zip(*batch))


def main():
    # Imposta il device (GPU se disponibile, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- STEP 1: Creazione delle pipeline di trasformazioni ----

    # Pipeline di training: augmentations con resize, flip, rotation e color jitter
    train_aug = get_train_augmentations(resize_size=800, p=0.5)
    train_transform = KorniaTransformWrapper(train_aug)

    # Pipeline di test: utilizza Kornia per applicare un semplice resize a 800x800
    test_aug = get_test_augmentations(resize_size=800)
    test_transform = KorniaTransformWrapper(test_aug)

    # ---- STEP 2: Creazione del dataset e split in train e test ----

    dataset = ChessboardCornersDataset(
        root="dataset/images",          # Percorso alle immagini
        csv_file="dataset/corners.csv",    # Percorso al file CSV con le annotazioni
        transforms=train_transform        # Applichiamo le trasformazioni di training per ora
    )

    # Split: ad esempio 80% training, 20% test
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Per il test set, assegniamo la pipeline di trasformazioni di test
    test_dataset.dataset.transforms = test_transform

    # ---- STEP 3: Creazione dei DataLoader ----

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # ---- STEP 4: Creazione del modello di Keypoint R-CNN ----

    def get_keypoint_model(num_keypoints=4):
        # Carica il modello pre-addestrato su COCO
        model = keypointrcnn_resnet50_fpn(weights="COCO_V1")
        # Sostituisci l'head per i keypoints con uno che predice num_keypoints (4 per la scacchiera)
        in_channels = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        model.roi_heads.keypoint_predictor = torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(
            in_channels=in_channels,
            num_keypoints=num_keypoints
        )
        return model

    model = get_keypoint_model(num_keypoints=4)
    model.to(device)

    # ---- STEP 5: Definizione dell'ottimizzatore e iperparametri ----

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10

    # ---- STEP 6: Training Loop ----

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            # Le immagini e i target sono liste (batch_size=1)
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # ---- STEP 7: Valutazione sul test set ----
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, targets in test_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                test_loss += losses.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}")

    # ---- STEP 8: Salvataggio del modello ----

    torch.save(model.state_dict(), "keypoint_rcnn_chessboard.pth")

if __name__ == '__main__':
    main()
