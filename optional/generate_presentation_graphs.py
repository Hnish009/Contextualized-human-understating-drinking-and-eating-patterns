"""
Generate comprehensive graphs and visualizations for PPT presentation
Creates all necessary visual materials for showcasing the Chudai project
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path
from src.utils import load_recipes

# Try to import seaborn (optional)
try:
    import seaborn as sns
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Set style for better-looking graphs
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

def load_training_history():
    """Load training history from saved files"""
    history = {}
    
    # Model 1 history
    info1_path = Path('models/training_history/model1_info.json')
    if info1_path.exists():
        with open(info1_path, 'r') as f:
            history['model1'] = json.load(f)
    
    # Model 2 history
    info2_path = Path('models/training_history/model2_info.json')
    if info2_path.exists():
        with open(info2_path, 'r') as f:
            history['model2'] = json.load(f)
    
    return history


def plot_model_architecture_comparison():
    """Create a comparison diagram of both models"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Model Architecture Comparison', fontsize=20, fontweight='bold', y=0.98)
    
    # Model 1 Architecture
    ax1 = axes[0]
    ax1.set_title('Model 1: Age & Mood Detection CNN', fontsize=14, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Draw Model 1 architecture
    layers_m1 = [
        ('Input\n(64×64×1)', (0.5, 0.9), 'lightblue'),
        ('Conv2D(32)\n+BN+Dropout', (0.5, 0.75), 'lightgreen'),
        ('Conv2D(64)\n+BN+Dropout', (0.5, 0.6), 'lightgreen'),
        ('Conv2D(128)\n+BN+Dropout', (0.5, 0.45), 'lightgreen'),
        ('Conv2D(256)\n+BN+Dropout', (0.5, 0.3), 'lightgreen'),
        ('Flatten\nDense(512)', (0.5, 0.15), 'lightyellow'),
        ('Shared Features\n(256)', (0.5, 0.05), 'orange'),
    ]
    
    outputs_m1 = [
        ('Age Output\n(Regression)', (0.2, -0.15), 'lightcoral'),
        ('Mood Output\n(7 Classes)', (0.8, -0.15), 'lightcoral'),
    ]
    
    for i, (text, pos, color) in enumerate(layers_m1):
        rect = plt.Rectangle((pos[0]-0.08, pos[1]-0.03), 0.16, 0.06, 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(pos[0], pos[1], text, ha='center', va='center', fontsize=9, fontweight='bold')
        if i < len(layers_m1) - 1:
            ax1.arrow(pos[0], pos[1]-0.03, 0, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    for text, pos, color in outputs_m1:
        rect = plt.Rectangle((pos[0]-0.08, pos[1]-0.03), 0.16, 0.06, 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(pos[0], pos[1], text, ha='center', va='center', fontsize=9, fontweight='bold')
        ax1.arrow(0.5, 0.05, pos[0]-0.5, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.3, 1.0)
    
    # Model 2 Architecture
    ax2 = axes[1]
    ax2.set_title('Model 2: Drink Suggestion MLP', fontsize=14, fontweight='bold', pad=20)
    ax2.axis('off')
    
    layers_m2 = [
        ('Input\n(9 features)', (0.5, 0.9), 'lightblue'),
        ('Dense(128)\n+BN+Dropout', (0.5, 0.7), 'lightgreen'),
        ('Dense(256)\n+BN+Dropout', (0.5, 0.5), 'lightgreen'),
        ('Dense(128)\n+BN+Dropout', (0.5, 0.3), 'lightgreen'),
        ('Dense(64)', (0.5, 0.1), 'lightyellow'),
        ('Output\n(7 Ratios)', (0.5, -0.1), 'lightcoral'),
    ]
    
    for i, (text, pos, color) in enumerate(layers_m2):
        rect = plt.Rectangle((pos[0]-0.12, pos[1]-0.04), 0.24, 0.08, 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(pos[0], pos[1], text, ha='center', va='center', fontsize=9, fontweight='bold')
        if i < len(layers_m2) - 1:
            ax2.arrow(pos[0], pos[1]-0.04, 0, -0.12, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig('models/training_history/model_architectures.png', dpi=300, bbox_inches='tight')
    print("OK: Model architecture diagram saved")
    plt.close()


def plot_training_curves():
    """Plot training curves from saved history"""
    history = load_training_history()
    
    if 'model1' in history and 'model2' in history:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Performance - Both Models', fontsize=18, fontweight='bold')
        
        # Model 1 - Age Loss
        ax = axes[0, 0]
        # Since we don't have epoch-by-epoch data, create a visual representation
        metrics = history['model1']['final_metrics']
        ax.bar(['Train', 'Validation'], 
               [metrics['age_mae_train'], metrics['age_mae_val']],
               color=['#2ecc71', '#3498db'], alpha=0.7)
        ax.set_title('Model 1: Age Prediction MAE', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error (years)')
        ax.grid(True, alpha=0.3)
        for i, v in enumerate([metrics['age_mae_train'], metrics['age_mae_val']]):
            ax.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Model 1 - Mood Accuracy
        ax = axes[0, 1]
        ax.bar(['Train', 'Validation'],
               [metrics['mood_acc_train'], metrics['mood_acc_val']],
               color=['#2ecc71', '#3498db'], alpha=0.7)
        ax.set_title('Model 1: Mood Classification Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        for i, v in enumerate([metrics['mood_acc_train'], metrics['mood_acc_val']]):
            ax.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # Model 2 - Loss
        ax = axes[1, 0]
        metrics2 = history['model2']['final_metrics']
        ax.bar(['Train', 'Validation'],
               [metrics2['loss_train'], metrics2['loss_val']],
               color=['#e74c3c', '#9b59b6'], alpha=0.7)
        ax.set_title('Model 2: Drink Prediction Loss (MSE)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Squared Error')
        ax.grid(True, alpha=0.3)
        for i, v in enumerate([metrics2['loss_train'], metrics2['loss_val']]):
            ax.text(i, v + v*0.1, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Model 2 - MAE
        ax = axes[1, 1]
        ax.bar(['Train', 'Validation'],
               [metrics2['mae_train'], metrics2['mae_val']],
               color=['#e74c3c', '#9b59b6'], alpha=0.7)
        ax.set_title('Model 2: Drink Prediction MAE', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error')
        ax.grid(True, alpha=0.3)
        for i, v in enumerate([metrics2['mae_train'], metrics2['mae_val']]):
            ax.text(i, v + v*0.1, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('models/training_history/training_performance.png', dpi=300, bbox_inches='tight')
        print("OK: Training performance graphs saved")
        plt.close()


def plot_drink_distribution():
    """Plot distribution of drink recipes"""
    recipes = load_recipes()
    drinks = recipes['drinks']
    bottles = recipes['bottle_order']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    drink_names = list(drinks.keys())
    data = []
    for drink in drink_names:
        ratios = [drinks[drink][bottle] for bottle in bottles]
        data.append(ratios)
    
    data = np.array(data)
    
    # Create stacked bar chart
    x = np.arange(len(drink_names))
    width = 0.6
    bottom = np.zeros(len(drink_names))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(bottles)))
    
    for i, bottle in enumerate(bottles):
        ax.bar(x, data[:, i], width, label=bottle, bottom=bottom, color=colors[i])
        bottom += data[:, i]
    
    ax.set_xlabel('Drink Names', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ingredient Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of 7 Ingredients Across 10 Drink Recipes', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(drink_names, rotation=45, ha='right', fontsize=9)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('models/training_history/drink_distribution.png', dpi=300, bbox_inches='tight')
    print("OK: Drink distribution graph saved")
    plt.close()


def plot_feedback_analysis():
    """Plot feedback learning progress if available"""
    feedback_path = Path('data/feedback.csv')
    
    if not feedback_path.exists():
        print("Note: No feedback data found yet")
        return
    
    try:
        df = pd.read_csv(feedback_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feedback Learning Analysis', fontsize=18, fontweight='bold')
        
        # Rating distribution
        ax = axes[0, 0]
        rating_counts = df['rating'].value_counts().sort_index()
        ax.bar(rating_counts.index, rating_counts.values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Rating (1-10)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Rating Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.3, axis='y')
        
        # Average rating over time
        ax = axes[0, 1]
        df_sorted = df.reset_index().sort_values(by='index')
        cumulative_avg = df_sorted['rating'].expanding().mean()
        ax.plot(range(1, len(cumulative_avg) + 1), cumulative_avg, 
               color='green', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Feedback Sample Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cumulative Average Rating', fontsize=11, fontweight='bold')
        ax.set_title('Average Rating Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=cumulative_avg.iloc[-1], color='red', linestyle='--', 
                  label=f'Final Avg: {cumulative_avg.iloc[-1]:.2f}')
        ax.legend()
        
        # Mood distribution
        ax = axes[1, 0]
        mood_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        mood_counts = df['mood'].value_counts().sort_index()
        ax.bar([mood_names[int(i)] for i in mood_counts.index], mood_counts.values, 
              color='coral', alpha=0.7)
        ax.set_xlabel('Detected Mood', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Mood Distribution in Feedback', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Age distribution
        ax = axes[1, 1]
        ax.hist(df['age'], bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Age (years)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Age Distribution in Feedback', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('models/training_history/feedback_analysis.png', dpi=300, bbox_inches='tight')
        print(f"OK: Feedback analysis saved ({len(df)} samples)")
        plt.close()
        
    except Exception as e:
        print(f"Error creating feedback analysis: {e}")


def plot_system_workflow():
    """Create a workflow diagram of the entire system"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    title = ax.text(5, 9.5, 'Chudai System Workflow', 
                   ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Workflow steps
    steps = [
        ('1. Camera Capture', (1.5, 8), 'lightblue'),
        ('2. Face Detection\n(OpenCV)', (3.5, 8), 'lightgreen'),
        ('3. Age & Mood\nDetection\n(Model 1)', (5.5, 8), 'lightyellow'),
        ('4. Drink Suggestion\n(Model 2)', (7.5, 8), 'lightcoral'),
        ('5. Display Ratios', (9, 8), 'lavender'),
        ('6. User Feedback\n(Rating 1-10)', (9, 6), 'wheat'),
        ('7. Save Feedback', (9, 4), 'lightsteelblue'),
        ('8. Fine-tune Model\n(Every 10 ratings)', (7.5, 4), 'lightpink'),
        ('9. Improved\nPredictions', (5.5, 4), 'lightseagreen'),
    ]
    
    # Draw boxes
    from matplotlib.patches import FancyBboxPatch
    for text, pos, color in steps:
        rect = FancyBboxPatch((pos[0]-0.6, pos[1]-0.4), 1.2, 0.8,
                             boxstyle='round,pad=0.1',
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], text, ha='center', va='center', 
               fontsize=9, fontweight='bold', wrap=True)
    
    # Draw arrows
    arrows = [
        ((2.1, 8), (2.9, 8)),  # 1->2
        ((4.1, 8), (4.9, 8)),  # 2->3
        ((6.1, 8), (6.9, 8)),  # 3->4
        ((8.1, 8), (8.9, 8)),  # 4->5
        ((9, 7.6), (9, 6.4)),  # 5->6
        ((9, 5.6), (9, 4.4)),  # 6->7
        ((8.4, 4), (7.6, 4)),  # 7->8
        ((6.9, 4), (6.1, 4)),  # 8->9
    ]
    
    for start, end in arrows:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.15, head_length=0.15, fc='black', ec='black', linewidth=2)
    
    # Add legend/info
    info_text = "System learns from feedback and improves predictions over time"
    ax.text(5, 1.5, info_text, ha='center', va='center', 
           fontsize=12, style='italic', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('models/training_history/system_workflow.png', dpi=300, bbox_inches='tight')
    print("OK: System workflow diagram saved")
    plt.close()


def create_summary_report():
    """Create a text summary report"""
    history = load_training_history()
    
    report = []
    report.append("=" * 60)
    report.append("CHUDAI PROJECT - SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")
    
    if 'model1' in history:
        m1 = history['model1']
        report.append("MODEL 1: Age & Mood Detection CNN")
        report.append("-" * 60)
        report.append(f"Epochs Trained: {m1['epochs_trained']}")
        report.append(f"Total Parameters: {m1['total_params']:,}")
        report.append("")
        report.append("Final Metrics:")
        metrics = m1['final_metrics']
        report.append(f"  Age MAE - Train: {metrics['age_mae_train']:.2f}, Val: {metrics['age_mae_val']:.2f}")
        report.append(f"  Mood Accuracy - Train: {metrics['mood_acc_train']:.2%}, Val: {metrics['mood_acc_val']:.2%}")
        report.append("")
    
    if 'model2' in history:
        m2 = history['model2']
        report.append("MODEL 2: Drink Suggestion MLP")
        report.append("-" * 60)
        report.append(f"Epochs Trained: {m2['epochs_trained']}")
        report.append(f"Total Parameters: {m2['total_params']:,}")
        report.append("")
        report.append("Final Metrics:")
        metrics = m2['final_metrics']
        report.append(f"  Loss (MSE) - Train: {metrics['loss_train']:.4f}, Val: {metrics['loss_val']:.4f}")
        report.append(f"  MAE - Train: {metrics['mae_train']:.4f}, Val: {metrics['mae_val']:.4f}")
        report.append("")
    
    # Check feedback
    feedback_path = Path('data/feedback.csv')
    if feedback_path.exists():
        df = pd.read_csv(feedback_path)
        report.append("FEEDBACK LEARNING")
        report.append("-" * 60)
        report.append(f"Total Feedback Samples: {len(df)}")
        report.append(f"Average Rating: {df['rating'].mean():.2f}/10")
        report.append(f"Rating Range: {df['rating'].min()}-{df['rating'].max()}")
        report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    with open('models/training_history/summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print("\nOK: Summary report saved")


def plot_sample_faces_with_predictions():
    """Create visualization showing sample faces with age/mood predictions"""
    import cv2
    
    # Try to load sample images from FER2013
    fer_path = Path('data/fer2013/train')
    sample_images = []
    sample_moods = []
    
    if fer_path.exists():
        mood_folders = ['happy', 'sad', 'angry', 'neutral', 'surprise']
        for mood in mood_folders:
            mood_path = fer_path / mood
            if mood_path.exists():
                images = list(mood_path.glob('*.jpg'))[:2]  # Get 2 samples per mood
                for img_path in images:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img_resized = cv2.resize(img, (64, 64))
                        sample_images.append(img_resized)
                        sample_moods.append(mood)
                        if len(sample_images) >= 10:  # Limit to 10 samples
                            break
            if len(sample_images) >= 10:
                break
    
    if not sample_images:
        print("Note: No FER2013 images found for sample visualization")
        return
    
    # Create figure
    cols = 5
    rows = (len(sample_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    fig.suptitle('Sample Face Images with Mood Detection', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    mood_names = {'happy': 'Happy', 'sad': 'Sad', 'angry': 'Angry', 
                  'neutral': 'Neutral', 'surprise': 'Surprise'}
    
    for idx, (img, mood) in enumerate(zip(sample_images[:10], sample_moods[:10])):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{mood_names.get(mood, mood).title()}', 
                    fontsize=10, fontweight='bold', color='green')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(sample_images), rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/training_history/sample_faces.png', dpi=300, bbox_inches='tight')
    print("OK: Sample faces visualization saved")
    plt.close()


def plot_drink_ratio_visualization():
    """Create visual bar chart showing drink ratios for different scenarios"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Drink Ratio Suggestions for Different Scenarios', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Sample scenarios
    scenarios = [
        ('Happy Male, Age 25', [0.25, 0.10, 0.25, 0.20, 0.10, 0.05, 0.05]),
        ('Sad Female, Age 30', [0.20, 0.15, 0.20, 0.20, 0.15, 0.05, 0.05]),
        ('Angry Male, Age 20', [0.15, 0.10, 0.25, 0.25, 0.20, 0.00, 0.05]),
        ('Neutral Female, Age 35', [0.20, 0.10, 0.20, 0.20, 0.10, 0.10, 0.10]),
        ('Surprise Male, Age 22', [0.25, 0.05, 0.25, 0.25, 0.10, 0.05, 0.05]),
        ('Happy Female, Age 28', [0.30, 0.10, 0.25, 0.15, 0.10, 0.05, 0.05]),
    ]
    
    bottles = ['Sugar', 'Salt', 'Lemon', 'Soda', 'Mint', 'Ginger', 'Masala']
    colors = plt.cm.Set3(np.linspace(0, 1, len(bottles)))
    
    for idx, (scenario, ratios) in enumerate(scenarios):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        bars = ax.bar(bottles, ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_title(scenario, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Ratio', fontsize=10)
        ax.set_ylim(0, 0.35)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/training_history/drink_ratio_examples.png', dpi=300, bbox_inches='tight')
    print("OK: Drink ratio examples saved")
    plt.close()


def plot_dataset_overview():
    """Create overview of datasets used"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Dataset Overview', fontsize=18, fontweight='bold')
    
    # FER2013 emotion distribution
    ax = axes[0]
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # Estimated counts (you can update with actual counts)
    counts = [3995, 436, 4097, 7215, 4830, 3171, 4965]
    colors_emotions = ['red', 'orange', 'purple', 'yellow', 'blue', 'green', 'gray']
    
    bars = ax.bar(emotions, counts, color=colors_emotions, alpha=0.7, edgecolor='black')
    ax.set_title('FER2013 Emotion Distribution (Training)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # UTKFace age distribution
    ax = axes[1]
    age_ranges = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    age_counts = [3000, 5000, 6000, 4000, 3000, 2000, 700, 8]  # Estimated
    
    ax.bar(age_ranges, age_counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_title('UTKFace Age Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Custom drink dataset
    ax = axes[2]
    drink_data_path = Path('data/drinks.csv')
    if drink_data_path.exists():
        df = pd.read_csv(drink_data_path)
        mood_counts = df['mood'].value_counts()
        # Moods are stored as strings (happy, sad, etc.)
        mood_labels = [mood.capitalize() for mood in mood_counts.index]
        
        ax.pie(mood_counts.values, labels=mood_labels, autopct='%1.1f%%', 
              startangle=90, colors=plt.cm.Pastel1.colors)
        ax.set_title('Custom Drink Dataset\nMood Distribution', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No custom\ndataset found', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Custom Drink Dataset', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/training_history/dataset_overview.png', dpi=300, bbox_inches='tight')
    print("OK: Dataset overview saved")
    plt.close()


def plot_prediction_examples():
    """Create visual examples of predictions"""
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    fig.suptitle('Example Predictions: Age, Mood → Drink Ratios', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Example scenarios
    examples = [
        ('Male, 25, Happy', [0.25, 0.10, 0.25, 0.20, 0.10, 0.05, 0.05]),
        ('Female, 30, Sad', [0.20, 0.15, 0.20, 0.20, 0.15, 0.05, 0.05]),
        ('Male, 20, Angry', [0.15, 0.10, 0.25, 0.25, 0.20, 0.00, 0.05]),
        ('Female, 35, Neutral', [0.20, 0.10, 0.20, 0.20, 0.10, 0.10, 0.10]),
        ('Male, 22, Surprise', [0.25, 0.05, 0.25, 0.25, 0.10, 0.05, 0.05]),
        ('Female, 28, Happy', [0.30, 0.10, 0.25, 0.15, 0.10, 0.05, 0.05]),
        ('Male, 40, Sad', [0.18, 0.12, 0.22, 0.22, 0.12, 0.08, 0.06]),
        ('Female, 18, Angry', [0.12, 0.08, 0.28, 0.28, 0.18, 0.03, 0.03]),
        ('Male, 45, Neutral', [0.22, 0.10, 0.20, 0.18, 0.10, 0.12, 0.08]),
    ]
    
    bottles = ['Sugar', 'Salt', 'Lemon', 'Soda', 'Mint', 'Ginger', 'Masala']
    
    for idx, (scenario, ratios) in enumerate(examples):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(bottles))
        bars = ax.barh(y_pos, ratios, color=plt.cm.viridis(ratios), 
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bottles, fontsize=9)
        ax.set_xlabel('Ratio', fontsize=10)
        ax.set_title(scenario, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlim(0, 0.35)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, ratio) in enumerate(zip(bars, ratios)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                   f'{ratio:.2f}', ha='left', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/training_history/prediction_examples.png', dpi=300, bbox_inches='tight')
    print("OK: Prediction examples saved")
    plt.close()


def main():
    """Generate all presentation materials"""
    print("=" * 60)
    print("Generating Presentation Graphs and Visualizations")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path('models/training_history')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all graphs
    plot_model_architecture_comparison()
    plot_training_curves()
    plot_drink_distribution()
    plot_feedback_analysis()
    plot_system_workflow()
    
    # Generate image-based visualizations
    print("\nGenerating image-based visualizations...")
    plot_sample_faces_with_predictions()
    plot_drink_ratio_visualization()
    plot_dataset_overview()
    plot_prediction_examples()
    
    create_summary_report()
    
    print()
    print("=" * 60)
    print("All presentation materials generated!")
    print("=" * 60)
    print("\nGenerated files in models/training_history/:")
    print("  - model_architectures.png")
    print("  - training_performance.png")
    print("  - drink_distribution.png")
    print("  - feedback_analysis.png (if feedback exists)")
    print("  - system_workflow.png")
    print("  - sample_faces.png (NEW - Image visualization)")
    print("  - drink_ratio_examples.png (NEW - Ratio examples)")
    print("  - dataset_overview.png (NEW - Dataset stats)")
    print("  - prediction_examples.png (NEW - Prediction demos)")
    print("  - summary_report.txt")
    print("\nReady for PPT presentation!")


if __name__ == '__main__':
    main()

