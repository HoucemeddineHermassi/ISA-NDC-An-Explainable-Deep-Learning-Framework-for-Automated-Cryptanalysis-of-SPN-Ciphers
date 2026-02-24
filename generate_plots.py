"""
Generate SVG plots for the paper using basic file operations.
Works around matplotlib installation issues.
"""
import json

def generate_training_plot():
    """Generate SVG plot for training history."""
    try:
        with open("training_history.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print("training_history.json not found. Run train.py first.")
        return
    
    epochs = list(range(1, len(history['train_acc']) + 1))
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    train_loss = history['train_loss']
    
    # Generate SVG manually
    width = 600
    height = 400
    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    
    # Scale values
    max_acc = max(max(train_acc), max(val_acc))
    min_acc = min(min(train_acc), min(val_acc))
    acc_range = max_acc - min_acc if max_acc > min_acc else 1
    
    def scale_y(val):
        return height - margin - ((val - min_acc) / acc_range) * plot_height
    
    def scale_x(i):
        return margin + (i / len(epochs)) * plot_width
    
    # Create point strings
    train_points = " ".join([f"{scale_x(i)},{scale_y(v)}" for i, v in enumerate(train_acc)])
    val_points = " ".join([f"{scale_x(i)},{scale_y(v)}" for i, v in enumerate(val_acc)])
    
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="white"/>
  
  <!-- Title -->
  <text x="{width/2}" y="30" text-anchor="middle" font-size="16" font-weight="bold">Training and Validation Accuracy</text>
  
  <!-- Axes -->
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black" stroke-width="2"/>
  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black" stroke-width="2"/>
  
  <!-- Y-axis labels -->
  <text x="{margin-10}" y="{margin}" text-anchor="end" font-size="12">{max_acc:.2f}</text>
  <text x="{margin-10}" y="{height-margin}" text-anchor="end" font-size="12">{min_acc:.2f}</text>
  <text x="20" y="{height/2}" text-anchor="middle" font-size="12" transform="rotate(-90, 20, {height/2})">Accuracy</text>
  
  <!-- X-axis labels -->
  <text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="12">Epoch</text>
  
  <!-- Grid lines -->
  <line x1="{margin}" y1="{height/2}" x2="{width-margin}" y2="{height/2}" stroke="#ddd" stroke-width="1" stroke-dasharray="4"/>
  
  <!-- Training Accuracy (Blue) -->
  <polyline points="{train_points}" fill="none" stroke="blue" stroke-width="2"/>
  
  <!-- Validation Accuracy (Red) -->
  <polyline points="{val_points}" fill="none" stroke="red" stroke-width="2"/>
  
  <!-- Legend -->
  <rect x="{width-140}" y="50" width="15" height="15" fill="blue"/>
  <text x="{width-120}" y="62" font-size="12">Train Acc</text>
  <rect x="{width-140}" y="75" width="15" height="15" fill="red"/>
  <text x="{width-120}" y="87" font-size="12">Val Acc</text>
</svg>'''
    
    with open("training_accuracy.svg", "w") as f:
        f.write(svg)
    print("Generated training_accuracy.svg")

def generate_attack_plot():
    """Generate SVG bar chart for attack results."""
    try:
        with open("attack_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("attack_results.json not found. Run attack.py first.")
        return
    
    width = 700
    height = 400
    margin = 80
    bar_width = 30
    
    # Categorize and colorize
    categories = []
    for label, score in results:
        if "True" in label:
            categories.append(('True Key', score, '#28a745'))
        elif "Near" in label:
            categories.append((label[:12], score, '#ffc107'))
        else:
            categories.append((label[:12], score, '#6c757d'))
    
    plot_width = width - 2 * margin
    bar_spacing = plot_width / len(categories)
    
    max_score = max(s for _, s, _ in categories)
    min_score = min(s for _, s, _ in categories)
    score_range = max_score - min_score if max_score > min_score else 0.1
    
    def scale_y(val):
        return height - margin - ((val - min_score) / score_range) * (height - 2*margin)
    
    bars_svg = ""
    labels_svg = ""
    for i, (label, score, color) in enumerate(categories):
        x = margin + i * bar_spacing + bar_spacing/4
        y = scale_y(score)
        bar_height = (height - margin) - y
        bars_svg += f'  <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}"/>\n'
        labels_svg += f'  <text x="{x + bar_width/2}" y="{height-margin+15}" text-anchor="middle" font-size="9" transform="rotate(45, {x + bar_width/2}, {height-margin+15})">{label}</text>\n'
    
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="white"/>
  
  <!-- Title -->
  <text x="{width/2}" y="30" text-anchor="middle" font-size="16" font-weight="bold">Key Recovery Attack: Candidate Scores</text>
  
  <!-- Axes -->
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black" stroke-width="2"/>
  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black" stroke-width="2"/>
  
  <!-- Y-axis labels -->
  <text x="{margin-10}" y="{margin}" text-anchor="end" font-size="11">{max_score:.3f}</text>
  <text x="{margin-10}" y="{height-margin}" text-anchor="end" font-size="11">{min_score:.3f}</text>
  <text x="20" y="{height/2}" text-anchor="middle" font-size="12" transform="rotate(-90, 20, {height/2})">Distinguisher Score</text>
  
  <!-- Bars -->
{bars_svg}
  <!-- Labels -->
{labels_svg}
  
  <!-- Legend -->
  <rect x="{width-130}" y="50" width="15" height="15" fill="#28a745"/>
  <text x="{width-110}" y="62" font-size="11">True Key</text>
  <rect x="{width-130}" y="70" width="15" height="15" fill="#ffc107"/>
  <text x="{width-110}" y="82" font-size="11">Near Keys</text>
  <rect x="{width-130}" y="90" width="15" height="15" fill="#6c757d"/>
  <text x="{width-110}" y="102" font-size="11">Random Keys</text>
</svg>'''
    
    with open("key_recovery_scores.svg", "w") as f:
        f.write(svg)
    print("Generated key_recovery_scores.svg")

if __name__ == "__main__":
    generate_training_plot()
    generate_attack_plot()
