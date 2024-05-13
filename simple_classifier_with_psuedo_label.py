from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from torchvision.transforms import v2, GaussianBlur
from tqdm import tqdm
import numpy as np
import time
from efficientnet_pytorch import EfficientNet
from nets_parts.datasets.psi_dataset_regions_cls import (
    PSIRegionClsDataset,
    PSIRegionClsDatasetParams,
)
from nets_parts.datasets.psi_dataset_random import (
    PSIRandomDataset,
    PSIRandomDatasetParams,
)
from nets_parts.nets_train_part import (
    run_valid, run_train,
    save_confusion_matrix, 
    get_data_iterator, get_data_iterator_pl 
)
from torchvision.transforms import RandAugment, ToPILImage
from torchvision.transforms.functional import pil_to_tensor
import gc
from nets_parts.datasets.psi_torch_dataset import TorchPSIDataset
from nets_parts.datasets.utils import low_entropy_filter
from torch.utils.tensorboard import SummaryWriter
import sys, json, os
from os.path import isdir, join
from nets_parts.datasets.utils import low_entropy_filter_single_input_with_prob
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
print(os.path.dirname(os.path.realpath(__file__)))
from nets_parts.RandAugment import RandAugment as MyRandAugment
THRESHOLD = 0.95

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
    batch_size_train,
    valid_image_num,
    checkpoint_path,
    cur_statistic_path,
    rand_aug_strange, 
    rand_aug_max_strange,
    n_ops,
    epoch_for_start_pl,
    use_all_data,
    writer,
    use_my_rand_aug,
    images_to_use,
    path_to_datasets
):
    global_train_loss = []
    global_train_acc = []
    global_valid_loss = []
    global_valid_acc = []
    best_train_acc = None
    best_valid_acc = None
    train_data_pseudo_label = None
    train_ds = None
    
    data_pl_iterator, train_data_pseudo_label = get_data_iterator_pl(
        path_to_data="/home/n.yakovlev/datasets/symblink/WSS2/train_valid",
        train_ds=train_data_pseudo_label,
        layer=layer,
        patch_size=patch_size,
        batch_size=64
    )
    generated_images: int = 0
    if use_my_rand_aug:
        RandAugmentator = MyRandAugment(ops_num=n_ops, cur_value=rand_aug_strange, max_value=rand_aug_max_strange)
    else:
        RandAugmentator = RandAugment(num_ops=n_ops, magnitude=rand_aug_strange, num_magnitude_bins=rand_aug_max_strange, fill=255)
    using_labels_in_cur_epoch = []
    generated_images_supervised = 0
    for epoch in range(n_epochs):
        model.train()
        if epoch % 5 == 0 or epoch == epoch_for_start_pl:
            generated_images_supervised = 0
            data_iterator, train_ds = get_data_iterator(
                path_to_data=path_to_datasets,
                train_ds=train_ds,
                layer=layer,
                patch_size=patch_size,
                batch_size= batch_size_train // 2 if epoch >= epoch_for_start_pl else batch_size_train 
            )
        print(f"Epoch {epoch}/{n_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        torch.set_grad_enabled(True)

        running_loss = 0.0
        running_corrects = 0
        using_labels_in_cur_epoch.append({k: int(0) for k in range(5)})

        # Iterate over data.
        for _ in tqdm(
            range(iterations_per_epoch), f"running epoch {epoch + 1}"
        ):
            model.eval()
            torch.set_grad_enabled(False)
            inputs_pl = None
            labels_pl = None
            cycle_try_to_get_pl = 0
            while epoch >= epoch_for_start_pl and (inputs_pl is None or inputs_pl.size(dim=0) < batch_size_train // 2):
                cycle_try_to_get_pl += 1
                gc.collect()
                torch.cuda.empty_cache()
                if inputs_pl is None and cycle_try_to_get_pl > 10:
                    break
                if generated_images > train_data_pseudo_label.__len__() - 10 * batch_size_train:
                    generated_images = 0
                    data_pl_iterator, train_data_pseudo_label = get_data_iterator_pl(
                        path_to_data="/home/n.yakovlev/datasets/symblink/WSS2/train_valid",
                        train_ds=train_data_pseudo_label,
                        layer=layer,
                        patch_size=patch_size,
                        batch_size=batch_size_train // 2 if epoch >= epoch_for_start_pl else batch_size_train 
                    )
                
                images_cur = next(data_pl_iterator).to(device)
                generated_images += images_cur.size(dim=0)
                labels_cur = model(images_cur)
                if (labels_cur.max(dim=1).values > THRESHOLD).sum().item() < 10:
                    continue
                # get pseudo labels with threshold
                images_cur = images_cur[labels_cur.max(dim=1).values > THRESHOLD]
                labels_cur = labels_cur[labels_cur.max(dim=1).values > THRESHOLD]
                labels_cur[labels_cur > THRESHOLD] = 1.0
                labels_cur[labels_cur < 1 - 1e-5] = 0
                if inputs_pl is None:
                    inputs_pl = images_cur
                    labels_pl = labels_cur
                else:
                    inputs_pl = torch.cat((inputs_pl, images_cur), dim=0)
                    labels_pl = torch.cat((labels_pl, labels_cur), dim=0)
            model.train()  # Set model to training mode
            torch.set_grad_enabled(True)
            if not (inputs_pl is None or labels_pl is None):
                inputs_pl, labels_pl = inputs_pl[:batch_size_train // 2], labels_pl[:batch_size_train // 2]
            if inputs_pl is None or labels_pl is None or inputs_pl.size(0) == 0:
                if generated_images_supervised > train_ds.__len__() - 8 * batch_size_train:
                    generated_images_supervised = 0
                    data_iterator, train_ds = get_data_iterator(
                        path_to_data=path_to_datasets,
                        train_ds=train_ds,
                        layer=layer,
                        patch_size=patch_size,
                        batch_size=64
                    )
                inputs, labels = next(data_iterator)
                generated_images_supervised += inputs.size(0)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # statistics
                preds = torch.argmax(outputs, dim=1)
                gts = torch.argmax(labels, dim=1)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == gts)
            else:
                uniq_values, uniq_counts = np.unique(labels_pl.argmax(1).cpu().numpy(), return_counts=True)
                for ind, i in zip(uniq_values, uniq_counts):
                    using_labels_in_cur_epoch[epoch][ind] += int(i)
                for image in inputs_pl:
                    if use_my_rand_aug:
                        temp_img = (image * 255).to(torch.uint8)
                        temp_img = ToPILImage()(temp_img)
                        temp_img = RandAugmentator(temp_img)
                        image = pil_to_tensor(temp_img) / 255
                    else:
                        image = (RandAugmentator((image * 255).to(torch.uint8))).to(torch.float32)
                        if image.max() > 1.1:
                            image /= 255
                # Each epoch has a training and validation phase
                if generated_images_supervised > train_ds.__len__() - 8 * batch_size_train:
                    generated_images_supervised = 0
                    data_iterator, train_ds = get_data_iterator(
                        path_to_data=path_to_datasets,
                        train_ds=train_ds,
                        layer=layer,
                        patch_size=patch_size,
                        batch_size=batch_size_train // 2
                    )
                inputs, labels = next(data_iterator)
                generated_images_supervised += inputs.size(0)
                labels_pl = labels_pl.to(labels.dtype)
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs_pl = inputs_pl.to(device)
                labels_pl = labels_pl.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                loss_l = criterion(outputs, labels)

                outputs_pl = model(inputs_pl)
                loss_u = criterion(outputs_pl, labels_pl)

                loss = loss_l + 1. / 7 * loss_u 
                loss.backward()
                optimizer.step()

                # statistics
                preds = torch.argmax(outputs, dim=1)
                gts = torch.argmax(labels, dim=1)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == gts)
            gc.collect()
            torch.cuda.empty_cache()

        if epoch >= 10:
            scheduler.step()

        epoch_loss = running_loss / iterations_per_epoch
        epoch_acc = running_corrects.float() / (
            iterations_per_epoch * inputs.size(0)
        )
        global_train_acc.append(float(epoch_acc))
        global_train_loss.append(float(epoch_loss))

        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        writer.add_scalar("Loss", epoch_loss, epoch)
        writer.add_scalar("Acc", epoch_acc, epoch)

        matrix_conf, epoch_loss, epoch_acc = run_valid(
            model=model,
            valid_loader=valid_loader,
            valid_images_num=valid_image_num,
            criterion=criterion,
            device=device
        )
        global_valid_acc.append(float(epoch_acc))
        global_valid_loss.append(float(epoch_loss))
        if best_valid_acc is None or best_valid_acc < epoch_acc:
            torch.save(model.state_dict(), f"{checkpoint_path}/best_valid_acc_best.pth")
            best_valid_acc = epoch_acc
        print(f"Val Loss: {epoch_loss:.4f} Val Acc: {epoch_acc:.4f}")
        writer.add_scalar("Loss valid", epoch_loss, epoch)
        writer.add_scalar("Accuracy valid", epoch_acc, epoch)
        if epoch % 1 == 0:
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
            with open(f"{cur_statistic_path}/using_pl.json", "w") as f:
                json.dump(
                    using_labels_in_cur_epoch,
                    f
                )
            path_to_save_conf_matrix = f"{cur_statistic_path}/confusion_matrix_best_epoch.png"
            save_confusion_matrix(matrix_conf, path_to_save_conf_matrix)
    train_data_pseudo_label = None
    train_ds = None
    if not (train_data_pseudo_label is None):
        train_data_pseudo_label.close()
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
        rand_aug_strange = int(parsed_json["rand_aug_strange"])
        rand_aug_max_strange = int(parsed_json["rand_aug_max_strange"])
        n_ops = int(parsed_json["n_ops"])
        device = f"cuda:{parsed_json["device"]}"
        epoch_for_start_pl = int(parsed_json["epoch_for_start_pl"])
        use_all_data = bool(parsed_json["use_all_data"])
        use_my_rand_aug = bool(parsed_json["use_my_rand_aug"])
        images_to_use = int(parsed_json["images_to_use"])
        path_to_datasets = parsed_json["path_to_datasets"]

        EXP_PATH = str(parsed_json["exprement_path"])
        os.makedirs(EXP_PATH, exist_ok=True)
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
                    model_ft = EfficientNet.from_pretrained(NN_NAME, num_classes=5)
                else:
                    model_ft = EfficientNet.from_name(NN_NAME, num_classes=5)
            else:
                model_ft = models.resnet50(weights="IMAGENET1K_V1")
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, 5)



    try:
        """
        path_to_checkpoint = parsed_json["path_to_starting_checkpoint"]
        pretrained_dict = torch.load(path_to_checkpoint)
        model_ft.load_state_dict(pretrained_dict)
        """
        if bool(parsed_json["using_checkpoint"]):
            pretrained_dict = torch.load(parsed_json["path_to_starting_checkpoint"])
            model_dict = model_ft.state_dict()

            processed_dict = {}

            for k in model_dict.keys():
                decomposed_key = k.split(".")
                if ("model" in decomposed_key):
                    pretrained_key = ".".join(decomposed_key[1:])
                    processed_dict[k] = pretrained_dict[pretrained_key]

            model_ft.load_state_dict(processed_dict, strict=False)
            print("Model weights loaded!")
    except Exception:
        print("Model weights doesn't loaded!")

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 1.2, 1, 1.2], device=device))

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARING_RATE)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=1, gamma=SCHEDULER_PARAM
    )

    data = np.load("/home/n.yakovlev/datasets/test_files_WSS2.npz")
    images = data["images"]
    valid_image_num = images.shape[0]
    labels = data["labels"]

    del data
    gc.collect()
    valid_dataset = TensorDataset(torch.tensor(images), torch.tensor(labels))
    del images, labels
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE_VALID, num_workers=8)

    train_model(
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        iterations_per_epoch=ITER_PER_EPOCH,  # len(train_ds) // batch_size,
        n_epochs=1500,
        valid_loader=valid_loader,
        patch_size=PATCH_SIZE,
        layer=LAYER, 
        batch_size_train=BATCH_SIZE_TRAIN,
        valid_image_num=valid_image_num,
        checkpoint_path=checkpoint_path,
        cur_statistic_path=cur_statistic_path,
        rand_aug_strange=rand_aug_strange,
        rand_aug_max_strange=rand_aug_max_strange,
        n_ops=n_ops,
        epoch_for_start_pl=epoch_for_start_pl,
        use_all_data=use_all_data,
        writer=writer,
        use_my_rand_aug=use_my_rand_aug,
        images_to_use=images_to_use,
        path_to_datasets=path_to_datasets
    )