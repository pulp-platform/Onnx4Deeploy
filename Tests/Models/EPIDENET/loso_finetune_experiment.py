"""
Leave-One-Subject-Out (LOSO) Cross-Subject Fine-tuning Experiment for EpiDeNet

Experimental Design:
1. For each subject i:
   - Train global model on subjects {1, 2, ..., N} \ {i}
   - Fine-tune on small amount of subject i's data
   - Evaluate on remaining subject i's data
2. Compare: No fine-tuning vs Fine-tuning with different sample sizes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from epidenet_model.epidenet import EpiDeNet
from utils.data_loading import load_real_data_new_new
import argparse


class LOSOFineTuneExperiment:
    def __init__(self,
                 data_path,
                 C=16,
                 T=1000,
                 N=11,
                 batch_size=256,
                 pretrain_epochs=200,
                 finetune_epochs=50,
                 pretrain_lr=0.001,
                 finetune_lr=0.0001,
                 method='matlab',
                 apply_notch_filter=False,
                 apply_bandpass_filter=True,
                 device='cuda'):

        self.data_path = data_path
        self.C = C
        self.T = T
        self.N = N
        self.batch_size = batch_size
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.method = method
        self.apply_notch_filter = apply_notch_filter
        self.apply_bandpass_filter = apply_bandpass_filter
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Get all subjects
        self.all_subjects = sorted(os.listdir(data_path))
        print(f"Found {len(self.all_subjects)} subjects: {self.all_subjects}")

    def load_subject_data(self, subjects):
        """Load data for specified subjects"""
        data_all = None
        label_all = None

        for subject in subjects:
            subject_path = os.path.join(self.data_path, subject)
            sessions = os.listdir(subject_path)

            for session in sessions:
                session_path = os.path.join(subject_path, session)
                data_new, label_new = load_real_data_new_new(
                    session_path,
                    sampling_freq=500,
                    window_size=2,
                    method=self.method,
                    zero_pad=False,
                    notch_filter=self.apply_notch_filter,
                    bandpass_filter=self.apply_bandpass_filter
                )

                if data_all is None:
                    data_all = data_new
                    label_all = label_new
                else:
                    data_all = np.concatenate((data_all, data_new), axis=2)
                    label_all = np.concatenate((label_all, label_new), axis=0)

        # Reshape: (channels, time, samples) -> (samples, 1, channels, time)
        data_all = np.swapaxes(data_all, 1, 2)  # (channels, samples, time)
        data_all = np.swapaxes(data_all, 0, 1)  # (samples, channels, time)
        data_all = np.expand_dims(data_all, axis=1)  # (samples, 1, channels, time)

        return data_all, label_all

    def train_model(self, model, train_loader, optimizer, criterion, epochs, desc="Training"):
        """Train model for specified epochs"""
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

            avg_loss = total_loss / len(train_loader)
            accuracy = 100.0 * correct / total

            if (epoch + 1) % 10 == 0:
                print(f"{desc} Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

        return model

    def evaluate_model(self, model, test_loader):
        """Evaluate model on test set"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        accuracy = 100.0 * correct / total
        return accuracy

    def run_loso_experiment(self, finetune_sample_sizes=[100, 200, 300, 400, 500]):
        """
        Run complete LOSO experiment

        Args:
            finetune_sample_sizes: List of sample sizes to use for fine-tuning
        """
        results = []

        for test_subject in self.all_subjects:
            print(f"\n{'='*80}")
            print(f"LOSO Experiment: Testing on subject {test_subject}")
            print(f"{'='*80}")

            # Step 1: Get train subjects (all except test subject)
            train_subjects = [s for s in self.all_subjects if s != test_subject]
            print(f"Training subjects: {train_subjects}")

            # Step 2: Load training data (N-1 subjects)
            print("Loading training data...")
            train_data, train_labels = self.load_subject_data(train_subjects)

            # Step 3: Load test subject data
            print(f"Loading test subject data: {test_subject}")
            test_subject_data, test_subject_labels = self.load_subject_data([test_subject])

            # Step 4: Split test subject data into finetune + test
            X_finetune_pool, X_test, y_finetune_pool, y_test = train_test_split(
                test_subject_data,
                test_subject_labels,
                test_size=0.5,  # Use 50% for testing
                random_state=42,
                stratify=test_subject_labels
            )

            print(f"Train data: {train_data.shape}")
            print(f"Finetune pool: {X_finetune_pool.shape}")
            print(f"Test data: {X_test.shape}")

            # Convert to tensors
            X_train_tensor = torch.tensor(train_data, dtype=torch.float32)
            y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            # Step 5: Train global model (pre-training)
            print("\n" + "="*50)
            print("Phase 1: Pre-training global model")
            print("="*50)

            pretrain_model = EpiDeNet(C=self.C, T=self.T, output_classes=self.N, p_dropout=0.0)
            pretrain_model = pretrain_model.to(self.device)

            optimizer = optim.Adam(pretrain_model.parameters(), lr=self.pretrain_lr)
            criterion = nn.CrossEntropyLoss()

            pretrain_model = self.train_model(
                pretrain_model,
                train_loader,
                optimizer,
                criterion,
                self.pretrain_epochs,
                desc="Pre-training"
            )

            # Step 6: Evaluate without fine-tuning
            print("\n" + "="*50)
            print("Evaluation: No Fine-tuning")
            print("="*50)

            no_finetune_acc = self.evaluate_model(pretrain_model, test_loader)
            print(f"Accuracy without fine-tuning: {no_finetune_acc:.2f}%")

            results.append({
                'subject': test_subject,
                'finetune_samples': 0,
                'accuracy': no_finetune_acc,
                'phase': 'no_finetune'
            })

            # Step 7: Fine-tune with different sample sizes
            for n_samples in finetune_sample_sizes:
                if n_samples > len(X_finetune_pool):
                    print(f"\nSkipping n_samples={n_samples} (not enough data)")
                    continue

                print("\n" + "="*50)
                print(f"Phase 2: Fine-tuning with {n_samples} samples")
                print("="*50)

                # Sample finetune data
                indices = np.random.choice(len(X_finetune_pool), size=n_samples, replace=False)
                X_finetune = X_finetune_pool[indices]
                y_finetune = y_finetune_pool[indices]

                X_finetune_tensor = torch.tensor(X_finetune, dtype=torch.float32)
                y_finetune_tensor = torch.tensor(y_finetune, dtype=torch.long)

                finetune_dataset = TensorDataset(X_finetune_tensor, y_finetune_tensor)
                finetune_loader = DataLoader(finetune_dataset, batch_size=min(self.batch_size, n_samples), shuffle=True)

                # Create new model and load pretrained weights
                finetune_model = EpiDeNet(C=self.C, T=self.T, output_classes=self.N, p_dropout=0.0)
                finetune_model.load_state_dict(pretrain_model.state_dict())
                finetune_model = finetune_model.to(self.device)

                # Fine-tune with lower learning rate
                optimizer_ft = optim.Adam(finetune_model.parameters(), lr=self.finetune_lr)

                finetune_model = self.train_model(
                    finetune_model,
                    finetune_loader,
                    optimizer_ft,
                    criterion,
                    self.finetune_epochs,
                    desc=f"Fine-tuning ({n_samples} samples)"
                )

                # Evaluate fine-tuned model
                finetune_acc = self.evaluate_model(finetune_model, test_loader)
                print(f"Accuracy after fine-tuning: {finetune_acc:.2f}%")
                print(f"Improvement: {finetune_acc - no_finetune_acc:+.2f}%")

                results.append({
                    'subject': test_subject,
                    'finetune_samples': n_samples,
                    'accuracy': finetune_acc,
                    'phase': 'finetune',
                    'improvement': finetune_acc - no_finetune_acc
                })

        return pd.DataFrame(results)

    def plot_results(self, results_df, save_path='loso_finetune_results.png'):
        """Plot experiment results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Accuracy vs Finetune Samples for each subject
        ax1 = axes[0]
        for subject in self.all_subjects:
            subject_data = results_df[results_df['subject'] == subject]
            ax1.plot(subject_data['finetune_samples'],
                    subject_data['accuracy'],
                    marker='o',
                    label=subject)

        ax1.set_xlabel('Number of Fine-tuning Samples')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('LOSO: Accuracy vs Fine-tuning Sample Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Average improvement
        ax2 = axes[1]
        avg_improvement = results_df[results_df['phase'] == 'finetune'].groupby('finetune_samples')['improvement'].mean()
        ax2.bar(avg_improvement.index, avg_improvement.values)
        ax2.set_xlabel('Number of Fine-tuning Samples')
        ax2.set_ylabel('Average Improvement (%)')
        ax2.set_title('Average Accuracy Improvement After Fine-tuning')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='LOSO Fine-tuning Experiment for EpiDeNet')
    parser.add_argument('--data_path', type=str, default='/scratch/thoriri/EOG_data/csv/',
                       help='Path to EOG data directory')
    parser.add_argument('--output_dir', type=str, default='loso_results',
                       help='Directory to save results')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                       help='Number of pre-training epochs')
    parser.add_argument('--finetune_epochs', type=int, default=30,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--pretrain_lr', type=float, default=0.001,
                       help='Pre-training learning rate')
    parser.add_argument('--finetune_lr', type=float, default=0.0001,
                       help='Fine-tuning learning rate')
    parser.add_argument('--finetune_samples', type=int, nargs='+',
                       default=[50, 100, 200, 300, 400, 500],
                       help='List of fine-tuning sample sizes to test')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize experiment
    experiment = LOSOFineTuneExperiment(
        data_path=args.data_path,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        pretrain_lr=args.pretrain_lr,
        finetune_lr=args.finetune_lr
    )

    # Run experiment
    print("Starting LOSO Fine-tuning Experiment...")
    print(f"Pre-training epochs: {args.pretrain_epochs}")
    print(f"Fine-tuning epochs: {args.finetune_epochs}")
    print(f"Fine-tuning sample sizes: {args.finetune_samples}")

    results_df = experiment.run_loso_experiment(finetune_sample_sizes=args.finetune_samples)

    # Save results
    results_path = os.path.join(args.output_dir, 'loso_finetune_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # No fine-tuning baseline
    no_ft = results_df[results_df['phase'] == 'no_finetune']
    print(f"\nBaseline (No Fine-tuning):")
    print(f"  Mean Accuracy: {no_ft['accuracy'].mean():.2f}%")
    print(f"  Std: {no_ft['accuracy'].std():.2f}%")

    # Fine-tuning results
    ft = results_df[results_df['phase'] == 'finetune']
    for n_samples in sorted(ft['finetune_samples'].unique()):
        ft_n = ft[ft['finetune_samples'] == n_samples]
        print(f"\nFine-tuning with {n_samples} samples:")
        print(f"  Mean Accuracy: {ft_n['accuracy'].mean():.2f}%")
        print(f"  Std: {ft_n['accuracy'].std():.2f}%")
        print(f"  Mean Improvement: {ft_n['improvement'].mean():+.2f}%")

    # Plot results
    plot_path = os.path.join(args.output_dir, 'loso_finetune_plot.png')
    experiment.plot_results(results_df, save_path=plot_path)

    print("\n" + "="*80)
    print("Experiment completed!")
    print("="*80)


if __name__ == '__main__':
    main()
