"""
Generate high-quality scientific SVG flowchart for ISA-NDC.
Zero dependencies.
"""

def generate_svg():
    width = 800
    height = 500
    
    # SVG Header
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="Arial, Helvetica, sans-serif">
  <rect width="100%" height="100%" fill="white"/>
  
  <!-- Definitions for markers -->
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="black" />
    </marker>
  </defs>
'''

    def rect(x, y, w, h, fill="#f8f9fa", stroke="black", stroke_width=1.5, rx=5, text=None, title=None):
        elem = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx}" />\n'
        if title:
            elem += f'<text x="{x + w/2}" y="{y + 20}" text-anchor="middle" font-weight="bold" font-size="12">{title}</text>\n'
            if text:
                lines = text.split('\n')
                start_y = y + h/2 + 5 - (len(lines)-1)*7
                for i, line in enumerate(lines):
                    elem += f'<text x="{x + w/2}" y="{start_y + i*14}" text-anchor="middle" font-size="11">{line}</text>\n'
        elif text:
            lines = text.split('\n')
            start_y = y + h/2 + 4 - (len(lines)-1)*6
            for i, line in enumerate(lines):
                elem += f'<text x="{x + w/2}" y="{start_y + i*14}" text-anchor="middle" font-size="11">{line}</text>\n'
        return elem

    def arrow(x1, y1, x2, y2):
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="1.5" marker-end="url(#arrow)" />\n'

    def connect_rects(r1, r2, side='right_to_left'):
        # r = (x, y, w, h)
        if side == 'right_to_left':
            return arrow(r1[0]+r1[2], r1[1]+r1[3]/2, r2[0], r2[1]+r2[3]/2)
        elif side == 'bottom_to_top':
            return arrow(r1[0]+r1[2]/2, r1[1]+r1[3], r2[0]+r2[2]/2, r2[1])
        return ""

    # === Phase 1: Image Analysis (Left) ===
    svg += '<rect x="20" y="20" width="220" height="460" fill="none" stroke="#666" stroke-dasharray="5,5" rx="10"/>\n'
    svg += '<text x="130" y="40" text-anchor="middle" font-weight="bold" fill="#666">Phase 1: Image Analysis</text>\n'
    
    r_img = (40, 60, 180, 50)
    svg += rect(*r_img, text="Medical Images\n(DICOM/X-ray)", fill="#e3f2fd")
    
    r_blocks = (40, 150, 180, 50)
    svg += rect(*r_blocks, text="Block Extraction\n(64-bit blocks)", fill="#e3f2fd")
    
    r_delta = (40, 240, 180, 50)
    svg += rect(*r_delta, text="Differential Analysis\n(Find Optimal Delta)", fill="#bbdefb")
    
    svg += connect_rects(r_img, r_blocks, 'bottom_to_top')
    svg += connect_rects(r_blocks, r_delta, 'bottom_to_top')

    # === Phase 2: Neural Distinguisher (Center) ===
    svg += '<rect x="280" y="20" width="240" height="460" fill="none" stroke="black" stroke-width="2" rx="10"/>\n'
    svg += '<text x="400" y="40" text-anchor="middle" font-weight="bold">Phase 2: Hybrid Neural Distinguisher</text>\n'
    
    r_input = (310, 70, 180, 40)
    svg += rect(*r_input, text="Input Difference\n(8x8 Grid)", fill="#f3e5f5")
    
    r_resnet = (310, 140, 180, 50)
    svg += rect(*r_resnet, text="ResNet Backbone\n(Local Features)", fill="#e1bee7")
    
    r_pos = (420, 200, 80, 30)
    svg += rect(*r_pos, text="Pos. Enc.", fill="#f8bbd0")
    
    r_vit = (310, 250, 180, 60)
    svg += rect(*r_vit, text="Vision Transformer\n(Global Attention)", fill="#ce93d8")
    
    r_mlp = (310, 350, 180, 40)
    svg += rect(*r_mlp, text="MLP Head\n(Sigmoid)", fill="#ba68c8")
    
    r_score = (310, 420, 180, 40)
    svg += rect(*r_score, text="Distinguisher Score\n(Probability)", fill="#ab47bc", stroke="black")
    
    # Internal connections
    svg += connect_rects(r_input, r_resnet, 'bottom_to_top')
    svg += connect_rects(r_resnet, r_vit, 'bottom_to_top')
    # Pos enc arrow
    svg += f'<line x1="{r_pos[0]+40}" y1="{r_pos[1]+30}" x2="{r_pos[0]+40}" y2="{r_vit[1]}" stroke="black" stroke-width="1" marker-end="url(#arrow)" />\n'
    svg += connect_rects(r_vit, r_mlp, 'bottom_to_top')
    svg += connect_rects(r_mlp, r_score, 'bottom_to_top')
    
    # Phase 1 -> Phase 2
    svg += arrow(r_delta[0]+r_delta[2], r_delta[1]+25, r_input[0], r_input[1]+20)
    svg += '<text x="265" y="160" text-anchor="middle" font-size="10" transform="rotate(-90, 265, 160)">Training Data</text>\n'

    # === Phase 3: Key Recovery (Right) ===
    svg += '<rect x="560" y="20" width="220" height="460" fill="none" stroke="#666" stroke-dasharray="5,5" rx="10"/>\n'
    svg += '<text x="670" y="40" text-anchor="middle" font-weight="bold" fill="#666">Phase 3: Key Recovery</text>\n'
    
    r_cipher = (580, 60, 180, 50)
    svg += rect(*r_cipher, text="Ciphertext Pairs\n(R rounds)", fill="#e8f5e9")
    
    r_guess = (580, 150, 180, 50)
    svg += rect(*r_guess, text="Key Guess K_g\n(Partial Decrypt)", fill="#c8e6c9")
    
    r_new_diff = (580, 240, 180, 50)
    svg += rect(*r_new_diff, text="New Difference\n(R-1 rounds)", fill="#a5d6a7")
    
    r_rank = (580, 420, 180, 40)
    svg += rect(*r_rank, text="Key Ranking\nArgmax(Score)", fill="#81c784")
    
    svg += connect_rects(r_cipher, r_guess, 'bottom_to_top')
    svg += connect_rects(r_guess, r_new_diff, 'bottom_to_top')
    
    # Distinguisher -> New Diff (Check)
    # Actually New Diff -> Distinguisher -> Score -> Rank
    
    # New Diff -> Distinguisher (Input)
    # Drawing a curve back to input is messy. Let's conceptually link Score to Rank.
    
    svg += arrow(r_score[0]+r_score[2], r_score[1]+20, r_rank[0], r_rank[1]+20)
    
    # New Diff triggers the loop
    svg += f'<path d="M{r_new_diff[0]} {r_new_diff[1]+25} Q 260 265 {r_input[0]} {r_input[1]+35}" fill="none" stroke="black" stroke-width="1.5" stroke-dasharray="4,4" marker-end="url(#arrow)"/>\n'
    svg += '<text x="500" y="300" text-anchor="middle" font-size="10">Query Model</text>\n'

    svg += '</svg>'
    
    with open("architecture_flowchart.svg", "w") as f:
        f.write(svg)
    print("Generated architecture_flowchart.svg")

if __name__ == "__main__":
    generate_svg()
