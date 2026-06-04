'''
# @Time    : 2025/5/14 9:34
# @Author  : Fengcheng Ji
# @Function: 模型推理，滑窗块提取 + 批量预测 + 不确定性图可视化
'''
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import median_filter
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import config


def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def predict_by_patches(model, ori_data, patch_size=5, device='cpu'):
    model.eval()
    padding_size = patch_size // 2
    padded = np.pad(ori_data, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant')
    H, W, C = ori_data.shape

    blocks = np.zeros((H * W, patch_size, patch_size, C))
    pbar = tqdm(total=H, desc="Data Processing")
    for i in range(H):
        row_patches = as_strided(
            padded[i:i + patch_size + padding_size * 2],
            shape=(W, patch_size, patch_size, C),
            strides=(padded.strides[1], padded.strides[0], padded.strides[1], padded.strides[2])
        )
        blocks[i * W:(i + 1) * W] = row_patches
        pbar.update(1)
    pbar.close()

    testData = torch.from_numpy(blocks).permute(0, 3, 1, 2)
    N = testData.size(0)
    test_predictions = []
    with torch.no_grad():
        bar = tqdm(total=(N + config.INFER_BATCH - 1) // config.INFER_BATCH, desc='Detection Processing', position=0, leave=True)
        for i in range(0, N, config.INFER_BATCH):
            batch = testData[i:i + config.INFER_BATCH].float().to(device, non_blocking=True)
            out = model(batch, inference=True)
            test_predictions.append(out.cpu())
            bar.update(1)
        bar.close()

    outputs_test = torch.cat(test_predictions, dim=0)
    uncertainty = torch.sigmoid(config.ALPHA * outputs_test)

    uncertainty_map0 = 2 * uncertainty.min(dim=1).values.cpu().numpy() - 1
    uncertainty_map = uncertainty_map0.reshape(H, W)

    plt.figure(figsize=(8, 6))
    plt.imshow(uncertainty_map)
    plt.colorbar(label='uncertainty')
    plt.title('uncertainty Map')
    plt.axis('off')
    plt.savefig('uncertainty_map.png', bbox_inches='tight', dpi=300)
    plt.show()

    binary_image = np.where(uncertainty_map > config.BIN_THRESH, 1, 0)
    filtered_uncertainty_map = median_filter(binary_image, size=config.FILTER_SIZE)
    plt.imshow(filtered_uncertainty_map, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.savefig('uncertainty_map0-1.png', bbox_inches='tight', dpi=300)
    plt.show()

    return filtered_uncertainty_map
