'''
# @Time    : 2025/5/14 9:34
# @Author  : Fengcheng Ji
# @Function: 主入口，数据加载、模型初始化、训练或推理、评估指标计算
'''
import time
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from utils import set_seed
from dataset import RemoteSensingDataset
from classifier import HyperbolicClassifier
from trainer import train
from inference import load_model, predict_by_patches
import config


if __name__ == "__main__":
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== 数据加载 ====================
    image_path = "./trainSample/trainData.npy"
    label_path = "./trainSample/label.npy"
    dataset = RemoteSensingDataset(image_path, label_path)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # ==================== 模型初始化 ====================
    model = HyperbolicClassifier(
        in_channels=dataset.in_channels,
        embed_dim=config.EMBED_DIM,
        num_classes=dataset.num_classes,
        queue_length=config.QUEUE_LENGTH,
        momentum=config.MOMENTUM,
        delta=config.DELTA,
        cc_weight=config.CC_WEIGHT,
        curvature=config.CURVATURE
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP, gamma=config.LR_GAMMA)

    # ==================== ====训练===========================
    #save_path = train(model, dataloader, optimizer, scheduler, device, epochs=config.EPOCHS)

    # ==================== 推理 ====================
    print("开始测试...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    save_path = './models/final.pth'
    model = load_model(model, save_path, device)
    test_image_path = "./trainSample/final.npy"
    img = np.load(test_image_path) / 10000
    pred_map = predict_by_patches(model, img, patch_size=config.PATCH_SIZE, device=device)

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"Training Peak Memory: {peak_memory:.4f} MB")
    end_time = time.time()
    print('整个场景的推理耗时为:' + str((end_time - start_time)))

    # ==================== 评估指标 ====================
    label_img = Image.open('testLabel/label.png').convert('L')
    label_array = np.array(label_img)
    labels = label_array.flatten()
    labels = (labels > 0).astype(int)
    uncertain_Scores = np.nan_to_num(pred_map, 0).flatten()

    cm = confusion_matrix(labels, uncertain_Scores, labels=[0, 1])
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / union
    miou = np.mean(iou)
    f1 = f1_score(labels, uncertain_Scores)
    precision = precision_score(labels, uncertain_Scores)
    recall = recall_score(labels, uncertain_Scores)

    anomaly_mask = (labels == 1)
    correct_anomaly = (uncertain_Scores == 1) & (labels == 1)
    oa_anomaly = correct_anomaly.sum() / anomaly_mask.sum()

    print(f"IoU for class 0 (normal): {iou[0]:.5f}")
    print(f"IoU for class 1 (abnormal): {iou[1]:.5f}")
    print(f"mIoU: {miou:.5}")
    print(f"F1 Score: {f1:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall:    {recall:.5f}")
    print(f"Overall Accuracy (OA): {oa_anomaly:.5f}")
