"""
Generate a high-quality scientific architecture flowchart for ISA-NDC.
Uses matplotlib to create a vector-quality diagram.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Styles
    box_props = dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1.5)
    arrow_props = dict(arrowstyle='-|>', fc='black', ec='black', lw=1.5)
    
    def add_box(x, y, width, height, text, title=None, color='#f0f0f0'):
        # Rectangle
        rect = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                              ec="black", fc=color, lw=1.5)
        ax.add_patch(rect)
        
        # Text
        cx = x + width/2
        cy = y + height/2
        if title:
            ax.text(cx, y + height - 0.2, title, ha='center', va='top', fontsize=10, fontweight='bold')
            ax.text(cx, y + height/2 - 0.2, text, ha='center', va='center', fontsize=8)
        else:
            ax.text(cx, cy, text, ha='center', va='center', fontsize=9)
            
        return (x, y, width, height)

    def connect(box1, box2, type='straight', text=None):
        # Center of box1
        x1 = box1[0] + box1[2]
        y1 = box1[1] + box1[3]/2
        
        # Center of box2 (left side)
        x2 = box2[0]
        y2 = box2[1] + box2[3]/2
        
        if type == 'straight':
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props)
        elif type == 'down_right':
            # From bottom of box1 to left of box2
            x1 = box1[0] + box1[2]/2
            y1 = box1[1]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                        arrowprops=dict(arrowstyle='-|>', connectionstyle="bar,angle=180,fraction=-0.2", fc='black', ec='black'))
            
        if text:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.1, text, ha='center', va='bottom', fontsize=8, style='italic')

    # === Components ===
    
    # 1. Image-Structure Analyzer (Top Left)
    # Container
    ax.add_patch(patches.Rectangle((0.5, 4.5), 3.5, 3.0, fill=False, linestyle='--', color='gray', lw=1))
    ax.text(0.6, 7.3, "Phase 1: Image-Structure Analyzer", fontsize=9, fontweight='bold', color='gray')
    
    b_images = add_box(1.0, 6.5, 2.5, 0.6, "Medical Image Dataset\n(DICOM/X-rays)")
    b_blocks = add_box(1.0, 5.5, 2.5, 0.6, "Block Extraction\n(64-bit blocks)")
    b_delta_analysis = add_box(1.0, 4.7, 2.5, 0.6, r"Differential Analysis\n(Find Optimal $\Delta_P$)")
    
    # Int connections
    ax.annotate("", xy=(2.25, 6.1), xytext=(2.25, 6.5), arrowprops=arrow_props)
    ax.annotate("", xy=(2.25, 5.3), xytext=(2.25, 5.5), arrowprops=arrow_props)
    
    # 2. Key Recovery / Data Prep (Bottom Left)
    # Container
    ax.add_patch(patches.Rectangle((0.5, 0.5), 3.5, 3.5, fill=False, linestyle='--', color='gray', lw=1))
    ax.text(0.6, 3.8, "Phase 2: Key Recovery Attack", fontsize=9, fontweight='bold', color='gray')
    
    b_pairs = add_box(1.0, 2.8, 2.5, 0.6, r"Ciphertext Pairs\n$(C_1, C_2)$")
    b_guess = add_box(1.0, 1.8, 2.5, 0.6, r"Partial Decryption\n(Guess $K_{last}$)")
    b_diff_c = add_box(1.0, 0.8, 2.5, 0.6, r"New Difference\n$\Delta C' = C'_1 \oplus C'_2$")
    
    # Int connections
    ax.annotate("", xy=(2.25, 2.4), xytext=(2.25, 2.8), arrowprops=arrow_props)
    ax.annotate("", xy=(2.25, 1.4), xytext=(2.25, 1.8), arrowprops=arrow_props)
    
    # 3. Neural Distinguisher (Right side)
    # Container
    ax.add_patch(patches.Rectangle((5.0, 1.0), 6.5, 6.0, fill=False, linestyle='-', color='black', lw=2))
    ax.text(5.2, 6.8, "Phase 3: Hybrid ViT-ResNet Distinguisher", fontsize=10, fontweight='bold')
    
    # Input
    b_input = add_box(5.5, 3.5, 1.0, 1.0, "Input\nGrid\n$8 \\times 8$")
    
    # ResNet
    b_resnet = add_box(7.0, 3.5, 1.5, 1.0, "ResNet Backbone\n(Local Features)", color='#e0efff')
    
    # Pos Enc
    b_pos = add_box(7.0, 5.0, 1.5, 0.6, "Positional\nEncoding", color='#fff0e0')
    
    # ViT
    b_vit = add_box(9.0, 3.0, 1.5, 2.0, "Vision\nTransformer\nEncoder\n(Global Attention)", color='#e0ffe0')
    
    # Head
    b_mlp = add_box(9.0, 1.5, 1.5, 0.8, "MLP Head\n(Sigmoid)", color='#ffe0e0')
    
    # Output score
    b_score = add_box(5.5, 1.5, 1.0, 0.8, "Score\n$p$", color='#ffffcc')
    
    # Connections Main
    
    # From Data Phase to Distinguisher
    # We need to show training vs inference.
    # Let's show Inference flow from bottom left
    ax.annotate("", xy=(5.5, 4.0), xytext=(3.5, 1.1), 
                arrowprops=dict(arrowstyle='-|>', connectionstyle="arc3,rad=-0.2", fc='black', ec='black', lw=1.5))
    ax.text(4.5, 2.8, "Candidates", rotation=35, fontsize=9)
    
    # From Top Left (Target Delta) to somewhere? Actually the target delta is implicit in training.
    # Let's show it feeding into "Training Data" conceptually, but maybe just link to the Input for clarity.
    ax.annotate("", xy=(5.5, 4.0), xytext=(3.5, 5.0), 
                arrowprops=dict(arrowstyle='-|>', connectionstyle="arc3,rad=0.2", fc='black', ec='black', lw=1.5))
    ax.text(4.5, 4.8, "Training Pairs", rotation=-25, fontsize=9)

    # Distinguisher Internal Flow
    connect(b_input, b_resnet)
    
    # Add Pos Encoding to path (summation)
    # ResNet -> Sum <- Pos
    # Then Sum -> ViT
    
    # Arrow from ResNet to ViT
    ax.annotate("", xy=(9.0, 4.0), xytext=(8.5, 4.0), arrowprops=arrow_props)
    
    # Arrow from Pos Enc down to the path
    ax.annotate("", xy=(7.75, 4.2), xytext=(7.75, 5.0), arrowprops=arrow_props)
    ax.text(7.8, 4.5, "+", fontsize=12, fontweight='bold')
    
    # ViT to MLP
    ax.annotate("", xy=(9.75, 2.3), xytext=(9.75, 3.0), arrowprops=arrow_props)
    
    # MLP to Score
    ax.annotate("", xy=(6.5, 1.9), xytext=(9.0, 1.9), arrowprops=arrow_props)
    
    # Feedback loop (Bayesian Optimization / Ranking)
    # Score -> Key Recovery
    ax.annotate("", xy=(2.25, 0.5), xytext=(6.0, 1.5), 
                arrowprops=dict(arrowstyle='-|>', connectionstyle="bar,angle=180,fraction=-0.2", fc='black', ec='black', ls='dashed'))
    ax.text(4.0, 0.3, "Update Rank $S(k_g)$", fontsize=9)

    plt.tight_layout()
    plt.savefig("architecture_flowchart.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("architecture_flowchart.png", format='png', dpi=300, bbox_inches='tight')
    print("Generated architecture_flowchart.pdf and .png")

if __name__ == "__main__":
    draw_flowchart()
