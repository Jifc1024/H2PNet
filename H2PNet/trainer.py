
import os
import torch


def train(model, dataloader, optimizer, scheduler, device, epochs=100, save_dir="models"):
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    iteration = 0
    save_path = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, loss, _ = model(x_batch, y_batch, iteration)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += x_batch.size(0)
            iteration += 1
        scheduler.step()
        acc = correct / total * 100
        avg_loss = total_loss / total
        print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%")

        if epoch == epochs:
            save_path = os.path.join(save_dir, f'final.pth')
            torch.save(model.state_dict(), save_path)
            print('model saved...............')

    return save_path
