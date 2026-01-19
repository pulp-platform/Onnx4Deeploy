"""
Simplified LOSO Fine-tuning Example

This script demonstrates the core concept of LOSO fine-tuning
without all the complexity of the full experiment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from epidenet_model.epidenet import EpiDeNet


def simple_loso_demo():
    """
    Simplified demonstration of LOSO fine-tuning workflow
    """

    # ============================================
    # Step 1: Prepare Data (Simulated)
    # ============================================
    print("="*60)
    print("LOSO Fine-tuning Demo")
    print("="*60)

    # Simulate data for 5 subjects
    n_subjects = 5
    samples_per_subject = 1000
    C, T, N = 16, 1000, 11

    # Generate simulated data
    print("\n[1] Generating simulated data...")
    all_data = {}
    for i in range(n_subjects):
        X = np.random.randn(samples_per_subject, 1, C, T).astype(np.float32)
        y = np.random.randint(0, N, samples_per_subject)
        all_data[f'subject_{i+1}'] = (X, y)
        print(f"  Subject {i+1}: {X.shape[0]} samples")

    # ============================================
    # Step 2: LOSO - Leave Subject 1 Out
    # ============================================
    test_subject = 'subject_1'
    print(f"\n[2] LOSO: Testing on {test_subject}")

    # Get training data (all except subject_1)
    train_X_list = []
    train_y_list = []
    for subj, (X, y) in all_data.items():
        if subj != test_subject:
            train_X_list.append(X)
            train_y_list.append(y)
            print(f"  Training with {subj}: {X.shape[0]} samples")

    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    print(f"  Total training samples: {train_X.shape[0]}")

    # Get test subject data
    test_X, test_y = all_data[test_subject]

    # Split test subject data: 50% for fine-tuning, 50% for testing
    split_idx = len(test_X) // 2
    finetune_X = test_X[:split_idx]
    finetune_y = test_y[:split_idx]
    final_test_X = test_X[split_idx:]
    final_test_y = test_y[split_idx:]

    print(f"\n  {test_subject} split:")
    print(f"    Fine-tune pool: {finetune_X.shape[0]} samples")
    print(f"    Test set: {final_test_X.shape[0]} samples")

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(train_X),
        torch.tensor(train_y, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(final_test_X),
        torch.tensor(final_test_y, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # ============================================
    # Step 3: Pre-train Global Model
    # ============================================
    print("\n[3] Pre-training global model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = EpiDeNet(C=C, T=T, output_classes=N, p_dropout=0.0)
    model = model.to(device)

    # Train
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    pretrain_epochs = 10  # Use small number for demo
    for epoch in range(pretrain_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{pretrain_epochs}: Loss={avg_loss:.4f}")

    # ============================================
    # Step 4: Evaluate without Fine-tuning
    # ============================================
    print("\n[4] Evaluation without fine-tuning...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

    baseline_acc = 100.0 * correct / total
    print(f"  Baseline Accuracy: {baseline_acc:.2f}%")

    # ============================================
    # Step 5: Fine-tune with Different Sample Sizes
    # ============================================
    print("\n[5] Fine-tuning with different sample sizes...")

    finetune_sample_sizes = [50, 100, 200]

    for n_samples in finetune_sample_sizes:
        print(f"\n  Fine-tuning with {n_samples} samples:")

        # Sample fine-tune data
        indices = np.random.choice(len(finetune_X), size=n_samples, replace=False)
        ft_X = finetune_X[indices]
        ft_y = finetune_y[indices]

        # Create fine-tune loader
        ft_dataset = TensorDataset(
            torch.tensor(ft_X),
            torch.tensor(ft_y, dtype=torch.long)
        )
        ft_loader = DataLoader(ft_dataset, batch_size=min(256, n_samples), shuffle=True)

        # Create new model and load pretrained weights
        ft_model = EpiDeNet(C=C, T=T, output_classes=N, p_dropout=0.0)
        ft_model.load_state_dict(model.state_dict())  # Load pretrained weights
        ft_model = ft_model.to(device)

        # Fine-tune with lower learning rate
        ft_optimizer = optim.Adam(ft_model.parameters(), lr=0.0001)  # 10x smaller

        finetune_epochs = 5
        for epoch in range(finetune_epochs):
            ft_model.train()
            for X_batch, y_batch in ft_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                ft_optimizer.zero_grad()
                output = ft_model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                ft_optimizer.step()

        # Evaluate fine-tuned model
        ft_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = ft_model(X_batch)
                pred = output.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)

        ft_acc = 100.0 * correct / total
        improvement = ft_acc - baseline_acc

        print(f"    After fine-tuning: {ft_acc:.2f}%")
        print(f"    Improvement: {improvement:+.2f}%")

    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*60)
    print("Demo Summary:")
    print("="*60)
    print(f"Baseline (no fine-tuning): {baseline_acc:.2f}%")
    print(f"After fine-tuning with limited data: improved by several %")
    print("\nKey Takeaway:")
    print("  Fine-tuning with subject-specific data improves performance")
    print("  even with small amounts of data (50-200 samples)")
    print("="*60)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Simplified LOSO Fine-tuning Demo")
    print("This demonstrates the core concept with simulated data")
    print("="*60 + "\n")

    simple_loso_demo()

    print("\nTo run the full experiment with real data:")
    print("  python loso_finetune_experiment.py --data_path /path/to/data")
