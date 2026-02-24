"""
Convert SVG to PDF using svglib.
"""
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

def convert():
    drawing = svg2rlg("architecture_flowchart.svg")
    renderPDF.drawToFile(drawing, "architecture_flowchart.pdf")
    print("Converted architecture_flowchart.svg to architecture_flowchart.pdf")

if __name__ == "__main__":
    convert()
