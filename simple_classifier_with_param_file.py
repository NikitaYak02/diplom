from pathlib import Path
import sys
print(sys.path)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from efficientnet_pytorch import EfficientNet
from nets_parts.datasets.psi_dataset_regions_cls import (
    PSIRegionClsDataset,
    PSIRegionClsDatasetParams,
)
from torch.utils.data import TensorDataset
import gc
from nets_parts.datasets.psi_torch_dataset import TorchPSIDataset
from torch.utils.tensorboard import SummaryWriter
import sys, json, os
from os.path import isdir, join
from PIL import Image
from nets_parts.nets_train_part import run_valid, save_confusion_matrix, get_data_iterator
from tqdm import tqdm
import time

def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    iterations_per_epoch,
    n_epochs,
    valid_loader,
    patch_size,
    layer,
    checkpoint_path,
    cur_statistic_path,
    valid_images_num,
    batch_size_train,
    images_to_use,
    path_to_datasets
):
    global_train_loss = []
    global_train_acc = []
    global_valid_loss = []
    global_valid_acc = []
    best_train_acc = None
    best_valid_acc = None
    train_ds = None
    for epoch in range(n_epochs):
        model.train()
        if epoch % images_to_use == 0:
            generated_images_supervised = 0
            data_iterator, train_ds = get_data_iterator(path_to_datasets, 
                                                        train_ds, 
                                                        layer=layer,
                                                        patch_size=patch_size,
                                                        batch_size=batch_size_train)
        print(f"Epoch {epoch}/{n_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        torch.set_grad_enabled(True)

        running_loss = []
        running_corrects = []

        # Iterate over data.
        for _ in tqdm(
            range(iterations_per_epoch), f"running epoch {epoch + 1}"
        ):
            np.random.seed(time.time_ns() % 2**32)
            if generated_images_supervised > train_ds.__len__() - batch_size_train - 2:
                generated_images_supervised = 0
                data_iterator, train_ds = get_data_iterator(
                    path_to_data=path_to_datasets,
                    train_ds=train_ds,
                    layer=layer,
                    patch_size=patch_size,
                    batch_size=batch_size_train
                )
            inputs, labels = next(data_iterator)
            generated_images_supervised += inputs.size(0)
            if inputs[0].mean() == inputs[1].mean() and labels[0][1] != 1.:
                print("\n\nSame image in train\n\n")
                exit(1)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            gts = torch.argmax(labels, dim=1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss.append(loss.item())
            running_corrects.append(torch.mean((preds == gts).to(torch.float)).item())

            gc.collect()
            torch.cuda.empty_cache()
    
        epoch_loss = np.array(running_loss).mean()
        epoch_acc = np.array(running_corrects).mean()

        if epoch >= 10:
            scheduler.step()
        
        global_train_acc.append(float(epoch_acc))
        global_train_loss.append(float(epoch_loss))
        if best_train_acc is None or best_train_acc < epoch_acc:
            torch.save(model.state_dict(), f"{checkpoint_path}/best_train_acc_{round(float(epoch_acc) * 100)}.pth")
            best_train_acc = epoch_acc

        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        writer.add_scalar("Loss", epoch_loss, epoch)
        writer.add_scalar("Acc", epoch_acc, epoch)
        
        #validate model and save needed stastic
        matrix_conf, epoch_loss, epoch_acc = run_valid(model, valid_loader, valid_images_num, criterion, device)
        writer.add_scalar("Loss valid", epoch_loss, epoch)
        writer.add_scalar("Accuracy valid", epoch_acc, epoch)
        print(f"Val Loss: {epoch_loss:.4f} Val Acc: {epoch_acc:.4f}")

        global_valid_acc.append(float(epoch_acc))
        global_valid_loss.append(float(epoch_loss))
        
        try:
            if best_valid_acc is None or best_valid_acc < epoch_acc:
                torch.save(model.state_dict(), f"{checkpoint_path}/best_valid_acc_{round(float(epoch_acc) * 100)}.pth")
                best_valid_acc = epoch_acc
                path_to_save_conf_matrix = f"{cur_statistic_path}/confusion_matrix_{epoch}.png"
                save_confusion_matrix(matrix_conf, path_to_save_conf_matrix)
            with open(f"{cur_statistic_path}/statistic.json", "w") as f:
                json.dump(
                    {
                        "train_loss": global_train_loss,
                        "train_acc": global_train_acc,
                        "valid_loss": global_valid_loss,
                        "valid_acc": global_valid_acc,
                    },
                    f
                )
        except Exception:
            print("\nError in saving!!!\n")

    if not (train_ds is None):
        train_ds.close()
    return model


if __name__ == "__main__":
    """
    {
        "batch_size_train": 192,
        "batch_size_valid": 512,
        "lr": 0.01,
        "exprement_path": "/home/n.yakovlev/my_best_program/diplom_8sem/experiments/classifier",

        "patch_size": 224,
        "layer": 2,
        "scheduler_param": 0.99,
        "pretrain": 1,

        "device": 1,

        "is_efficientnet": 1
        "nn_name": "efficientnet-b4"
        "iter_per_epoch": 26
    }
    """
    path_to_params = sys.argv[1]
    with open(path_to_params) as f:
        parsed_file = f.read()
        parsed_json = json.loads(parsed_file)
        BATCH_SIZE_TRAIN = int(parsed_json["batch_size_train"])
        BATCH_SIZE_VALID = int(parsed_json["batch_size_valid"])
        LEARING_RATE = float(parsed_json["lr"])
        LAYER = int(parsed_json["layer"])

        PRETRAIN = bool(parsed_json["pretrain"])
        EXP_PATH = str(parsed_json["exprement_path"])
        os.makedirs(EXP_PATH, exist_ok=True)
        PATCH_SIZE = int(parsed_json["patch_size"])
        SCHEDULER_PARAM = float(parsed_json["scheduler_param"])

        IS_EFFICIENTNET = bool(parsed_json["is_efficientnet"])
        NN_NAME = str(parsed_json["nn_name"])
        ITER_PER_EPOCH = int(parsed_json["iter_per_epoch"])
        device = f"cuda:{parsed_json["device"]}"
        images_to_use = int(parsed_json["images_to_use"])
        path_to_datasets = parsed_json["path_to_datasets"]

        onlydirs = [f for f in os.listdir(EXP_PATH) if isdir(join(EXP_PATH, f))]
        cur_exp_path = f"{EXP_PATH}/{len(onlydirs)}"
        os.makedirs(cur_exp_path)
        cur_statistic_path = f"{cur_exp_path}/statistic"
        os.makedirs(cur_statistic_path)
        checkpoint_path = cur_exp_path + "/checkpoints"
        os.makedirs(checkpoint_path)
        writer = SummaryWriter(f'{cur_exp_path}/tensorboard_logs')
        with open(f"{cur_exp_path}/params.json", 'w') as f:
            json.dump(parsed_json, f)

    if IS_EFFICIENTNET:
        if PRETRAIN:
            model = EfficientNet.from_pretrained(NN_NAME, num_classes=5)
        else:
            model = EfficientNet.from_name(NN_NAME, num_classes=5)
    else:
        model = models.resnet50(weights="IMAGENET1K_V1")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)

    try:
        """
        path_to_checkpoint = parsed_json["path_to_starting_checkpoint"]
        pretrained_dict = torch.load(path_to_checkpoint)
        model_ft.load_state_dict(pretrained_dict)
        """
        if bool(parsed_json["using_checkpoint"]):
            pretrained_dict = torch.load(parsed_json["path_to_starting_checkpoint"])
            model_dict = model.state_dict()

            processed_dict = {}

            for k in model_dict.keys():
                decomposed_key = k.split(".")
                if ("model" in decomposed_key):
                    pretrained_key = ".".join(decomposed_key[1:])
                    processed_dict[k] = pretrained_dict[pretrained_key]

            model.load_state_dict(processed_dict, strict=False)

    except Exception:
        print("Model weights doesn't loaded!")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=LEARING_RATE)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=SCHEDULER_PARAM
    )

    data = np.load("/home/n.yakovlev/datasets/test_files_WSS2.npz")
    images = data["images"]
    labels = data["labels"]
    valid_images_num = labels.shape[0]
    del data
    gc.collect()
    valid_dataset = TensorDataset(torch.tensor(images), torch.tensor(labels))
    del images, labels
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE_VALID, num_workers=8)
    
    train_model(
        model,
        criterion,
        optimizer,
        exp_lr_scheduler,
        iterations_per_epoch=ITER_PER_EPOCH,  # len(train_ds) // batch_size,
        n_epochs=100,
        valid_loader=valid_loader,
        patch_size=PATCH_SIZE,
        layer=LAYER, 
        checkpoint_path=checkpoint_path,
        cur_statistic_path=cur_statistic_path,
        valid_images_num=valid_images_num,
        batch_size_train=BATCH_SIZE_TRAIN,
        images_to_use=images_to_use,
        path_to_datasets=path_to_datasets
    )