"""
Generate detailed training curves and plots from CSV logs
Creates comprehensive curve visualizations for presentation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')


def plot_detailed_training_curves():
    """Plot detailed training curves from CSV logs"""
    
    # Model 1 curves
    csv1_path = Path('models/training_history/model1_training_progress.csv')
    
    if csv1_path.exists():
        df1 = pd.read_csv(csv1_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model 1: Age & Mood Detection - Detailed Training Curves', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        epochs = range(1, len(df1) + 1)
        
        # Age Loss
        ax = axes[0, 0]
        ax.plot(epochs, df1['age_output_loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, df1['val_age_output_loss'], 'r--', label='Validation Loss', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Age Loss (MAE)', fontsize=12, fontweight='bold')
        ax.set_title('Age Prediction Loss Over Epochs', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Age MAE
        ax = axes[0, 1]
        ax.plot(epochs, df1['age_output_mae'], 'b-', label='Training MAE', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, df1['val_age_output_mae'], 'r--', label='Validation MAE', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error (years)', fontsize=12, fontweight='bold')
        ax.set_title('Age Prediction MAE Over Epochs', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Mood Loss
        ax = axes[1, 0]
        ax.plot(epochs, df1['mood_output_loss'], 'g-', label='Training Loss', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, df1['val_mood_output_loss'], 'orange', linestyle='--', label='Validation Loss', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mood Loss (Crossentropy)', fontsize=12, fontweight='bold')
        ax.set_title('Mood Classification Loss Over Epochs', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Mood Accuracy
        ax = axes[1, 1]
        ax.plot(epochs, df1['mood_output_accuracy'], 'g-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, df1['val_mood_output_accuracy'], 'orange', linestyle='--', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Mood Classification Accuracy Over Epochs', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history/model1_detailed_curves.png', dpi=300, bbox_inches='tight')
        print("OK: Model 1 detailed curves saved")
        plt.close()
    else:
        print("Note: Model 1 training progress CSV not found")
    
    # Model 2 curves
    csv2_path = Path('models/training_history/model2_training_progress.csv')
    
    if csv2_path.exists():
        df2 = pd.read_csv(csv2_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Model 2: Drink Suggestion - Detailed Training Curves', 
                    fontsize=18, fontweight='bold')
        
        epochs = range(1, len(df2) + 1)
        
        # Loss
        ax = axes[0]
        ax.plot(epochs, df2['loss'], 'purple', label='Training Loss', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, df2['val_loss'], 'red', linestyle='--', label='Validation Loss', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        ax.set_title('Loss Over Epochs', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # MAE
        ax = axes[1]
        ax.plot(epochs, df2['mae'], 'purple', label='Training MAE', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, df2['val_mae'], 'red', linestyle='--', label='Validation MAE', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
        ax.set_title('MAE Over Epochs', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history/model2_detailed_curves.png', dpi=300, bbox_inches='tight')
        print("OK: Model 2 detailed curves saved")
        plt.close()
    else:
        print("Note: Model 2 training progress CSV not found")


def plot_combined_learning_curves():
    """Create combined learning curves showing both models"""
    
    # Load training history from JSON
    history = {}
    info1_path = Path('models/training_history/model1_info.json')
    info2_path = Path('models/training_history/model2_info.json')
    
    if info1_path.exists():
        with open(info1_path, 'r') as f:
            history['model1'] = json.load(f)
    
    if info2_path.exists():
        with open(info2_path, 'r') as f:
            history['model2'] = json.load(f)
    
    if 'model1' in history and 'model2' in history:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Combined Learning Curves - Model Comparison', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        m1_metrics = history['model1']['final_metrics']
        m2_metrics = history['model2']['final_metrics']
        
        # Model 1 Metrics
        ax = axes[0, 0]
        categories = ['Age MAE\nTrain', 'Age MAE\nVal', 'Mood Acc\nTrain', 'Mood Acc\nVal']
        values = [m1_metrics['age_mae_train'], m1_metrics['age_mae_val'],
                 m1_metrics['mood_acc_train'], m1_metrics['mood_acc_val']]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_title('Model 1: Final Metrics', fontsize=13, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if val < 1:
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Model 2 Metrics
        ax = axes[0, 1]
        categories = ['Loss\nTrain', 'Loss\nVal', 'MAE\nTrain', 'MAE\nVal']
        values = [m2_metrics['loss_train'], m2_metrics['loss_val'],
                 m2_metrics['mae_train'], m2_metrics['mae_val']]
        colors = ['#9b59b6', '#e74c3c', '#3498db', '#f39c12']
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_title('Model 2: Final Metrics', fontsize=13, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.1,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Training Progress Comparison
        ax = axes[1, 0]
        epochs_m1 = history['model1']['epochs_trained']
        epochs_m2 = history['model2']['epochs_trained']
        models = ['Model 1\n(CNN)', 'Model 2\n(MLP)']
        epochs = [epochs_m1, epochs_m2]
        colors = ['#3498db', '#9b59b6']
        bars = ax.bar(models, epochs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_title('Training Epochs Comparison', fontsize=13, fontweight='bold')
        ax.set_ylabel('Epochs', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, epoch in zip(bars, epochs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                   f'{epoch}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Model Size Comparison
        ax = axes[1, 1]
        params_m1 = history['model1']['total_params']
        params_m2 = history['model2']['total_params']
        models = ['Model 1\n(CNN)', 'Model 2\n(MLP)']
        params = [params_m1 / 1e6, params_m2 / 1e6]  # Convert to millions
        colors = ['#e74c3c', '#2ecc71']
        bars = ax.bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_title('Model Size Comparison', fontsize=13, fontweight='bold')
        ax.set_ylabel('Parameters (Millions)', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                   f'{param:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('models/training_history/combined_learning_curves.png', dpi=300, bbox_inches='tight')
        print("OK: Combined learning curves saved")
        plt.close()


def plot_feedback_learning_curve():
    """Plot feedback learning progress over time"""
    feedback_path = Path('data/feedback.csv')
    
    if not feedback_path.exists():
        print("Note: No feedback data for learning curve")
        return
    
    try:
        df = pd.read_csv(feedback_path)
        
        if len(df) < 2:
            print("Note: Need at least 2 feedback samples for learning curve")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feedback Learning Progress', fontsize=18, fontweight='bold', y=0.98)
        
        # Rating over time
        ax = axes[0, 0]
        sample_numbers = range(1, len(df) + 1)
        ax.plot(sample_numbers, df['rating'], 'o-', color='green', linewidth=2, markersize=6, label='Ratings')
        ax.axhline(y=df['rating'].mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Average: {df["rating"].mean():.2f}')
        ax.set_xlabel('Feedback Sample Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rating (1-10)', fontsize=12, fontweight='bold')
        ax.set_title('Rating Over Time', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 11)
        ax.set_xticks(sample_numbers)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Cumulative average
        ax = axes[0, 1]
        cumulative_avg = df['rating'].expanding().mean()
        ax.plot(sample_numbers, cumulative_avg, 'b-', linewidth=2, marker='o', markersize=4)
        ax.fill_between(sample_numbers, cumulative_avg, alpha=0.3, color='blue')
        ax.set_xlabel('Feedback Sample Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Average Rating', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Average Rating Trend', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 11)
        ax.grid(True, alpha=0.3)
        
        # Rating distribution histogram
        ax = axes[1, 0]
        ax.hist(df['rating'], bins=range(1, 12), color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Rating', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Rating Distribution', fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.3, axis='y')
        
        # Moving average (if enough data)
        ax = axes[1, 1]
        if len(df) >= 5:
            window = min(5, len(df) // 2)
            moving_avg = df['rating'].rolling(window=window).mean()
            ax.plot(sample_numbers, df['rating'], 'o', color='lightblue', markersize=4, label='Ratings', alpha=0.5)
            ax.plot(sample_numbers[window-1:], moving_avg[window-1:], 'r-', linewidth=2, 
                   label=f'Moving Average ({window})')
            ax.set_xlabel('Feedback Sample Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Rating', fontsize=12, fontweight='bold')
            ax.set_title('Ratings with Moving Average', fontsize=13, fontweight='bold')
            ax.set_ylim(0, 11)
            ax.legend(fontsize=11)
        else:
            ax.plot(sample_numbers, df['rating'], 'o-', color='green', linewidth=2, markersize=8)
            ax.set_xlabel('Feedback Sample Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Rating', fontsize=12, fontweight='bold')
            ax.set_title('Rating Trend', fontsize=13, fontweight='bold')
            ax.set_ylim(0, 11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history/feedback_learning_curve.png', dpi=300, bbox_inches='tight')
        print(f"OK: Feedback learning curve saved ({len(df)} samples)")
        plt.close()
        
    except Exception as e:
        print(f"Error creating feedback curve: {e}")


def main():
    """Generate all detailed curves"""
    print("=" * 60)
    print("Generating Detailed Training Curves and Plots")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path('models/training_history')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate curves
    plot_detailed_training_curves()
    plot_combined_learning_curves()
    plot_feedback_learning_curve()
    
    print()
    print("=" * 60)
    print("All detailed curves generated!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - model1_detailed_curves.png (if CSV exists)")
    print("  - model2_detailed_curves.png (if CSV exists)")
    print("  - combined_learning_curves.png")
    print("  - feedback_learning_curve.png (if feedback exists)")
    print("\nReady for PPT presentation!")


if __name__ == '__main__':
    main()

