"""Regenerate the SVG and PNG README figures (requires Pillow)."""

from pathlib import Path
import csv
import math
from html import escape
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
INK, MUTED, LINE = "#183447", "#516779", "#a6bac8"
TEAL, SOFT, ORANGE = "#087f8c", "#edf7f8", "#fff3e4"


class Figure:
    def __init__(self, width, height, title):
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>{escape(title)}</title><rect width="100%" height="100%" fill="white"/>']

    def text(self, x, y, value, size=23, color=INK, bold=False, center=False):
        font = ImageFont.truetype(str(BOLD if bold else FONT), size)
        for index, line in enumerate(value.split("\n")):
            yy = y + index * (size + 10)
            self.draw.text((x, yy), line, font=font, fill=color, anchor="mm" if center else "lm")
            self.svg.append(f'<text x="{x}" y="{yy}" dominant-baseline="central" text-anchor="{"middle" if center else "start"}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(line)}</text>')

    def box(self, x, y, w, h, fill=SOFT, stroke=LINE):
        self.draw.rounded_rectangle((x, y, x+w, y+h), radius=16, fill=fill, outline=stroke, width=2)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def arrow(self, points, color=LINE):
        self.draw.line(points, fill=color, width=3)
        (ax, ay), (bx, by) = points[-2:]
        angle = math.atan2(by-ay, bx-ax)
        triangle = [(bx, by), (bx-13*math.cos(angle-.45), by-13*math.sin(angle-.45)), (bx-13*math.cos(angle+.45), by-13*math.sin(angle+.45))]
        self.draw.polygon(triangle, fill=color)
        self.svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        self.svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in triangle)}" fill="{color}"/>')

    def save(self, name):
        (OUT/f"{name}.svg").write_text("\n".join(self.svg)+"\n</svg>\n")
        self.image.save(OUT/f"{name}.png")


def workflow():
    f = Figure(1440, 1150, "Audio-visual fusion: transcript filtering and missing-feature handling")
    f.text(60, 57, "Audio-Visual Fusion with Missing Modalities", 39, bold=True)
    f.text(60, 111, "Filter-and-Concat: filter unreliable text, retain available features, then fuse.", 25, MUTED)
    for x, title, subtitle in [(60, "AUDIO", "Interview recording"), (520, "VIDEO", "Interview frames"), (980, "TEXT FROM AUDIO", "Whisper + question-aware prompt")]:
        f.box(x, 170, 400, 100, fill="#f4f7fa")
        f.text(x+200, 205, title, 24, bold=True, center=True)
        f.text(x+200, 242, subtitle, 20, MUTED, center=True)
    f.arrow([(260,270),(260,330)])
    f.arrow([(720,270),(720,330)])
    f.arrow([(1180,270),(1180,320)])
    f.box(60,330,400,125)
    f.text(260,365,"wav2vec audio features",25,bold=True,center=True)
    f.text(260,408,"Pool / align sequence features",21,MUTED,center=True)
    f.box(520,330,400,125)
    f.text(720,365,"FaceNet face crop",25,bold=True,center=True)
    f.text(720,408,"CLIP video features",23,MUTED,center=True)
    f.box(980,320,400,180,fill="#fff8eb",stroke="#dab771")
    f.text(1180,350,"TRANSCRIPT FILTER",25,bold=True,center=True)
    f.text(1000,395,"Empty / too long / blacklist?",21)
    f.text(1000,430,"Low or unavailable confidence?",21)
    f.text(1000,465,"Keep only answers that pass",21,MUTED)
    f.arrow([(260,455),(260,530)])
    f.arrow([(720,455),(720,530)])
    f.box(60,530,400,105,fill="#f4f7fa")
    f.text(260,560,"Retain available audio",24,bold=True,center=True)
    f.text(260,603,"even when video / text is absent",21,MUTED,center=True)
    f.box(520,530,400,105,fill=ORANGE,stroke="#dab771")
    f.text(720,560,"Failed face detection?",24,bold=True,center=True)
    f.text(720,603,"Mark video as missing",22,MUTED,center=True)
    f.arrow([(1180,500),(1180,523),(1075,523),(1075,550)],color=TEAL)
    f.arrow([(1180,523),(1285,523),(1285,550)],color="#bd8240")
    f.box(980,550,190,115,fill="#e9f5ee",stroke="#82b49a")
    f.text(1075,579,"Accepted",23,bold=True,center=True)
    f.text(1075,620,"CLIP text\nencoding",19,MUTED,center=True)
    f.box(1190,550,190,115,fill=ORANGE,stroke="#dab771")
    f.text(1285,579,"Rejected",23,bold=True,center=True)
    f.text(1285,620,"Empty text\nmissing = 1",19,MUTED,center=True)
    for x,y in [(260,635),(720,635),(1075,665),(1285,665)]:
        f.arrow([(x,y),(x,710)])
    f.box(60,710,1320,100,fill="#e7f2f7",stroke="#83adbf")
    f.text(720,745,"Keep explicit missing flags; zero-fill absent feature blocks",28,bold=True,center=True)
    f.text(720,785,"1 = missing, 0 = present. Each available modality stays in the sample.",23,MUTED,center=True)
    f.arrow([(720,810),(720,860)],color=TEAL)
    f.box(60,860,1320,95,fill=SOFT,stroke=TEAL)
    f.text(720,892,"CONCATENATE",25,color=TEAL,bold=True,center=True)
    f.text(720,930,"[ audio | video | text | missing_A | missing_V | missing_T ]",26,center=True)
    f.arrow([(720,955),(720,1000)],color=TEAL)
    f.box(470,1000,500,85,fill=TEAL,stroke=TEAL)
    f.text(720,1042,"MLP  →  Binary prediction",29,color="white",bold=True,center=True)
    f.text(60,1120,"Workflow from experiment notes; feature-level reference implementation provided.",21,MUTED)
    f.save("filtering-workflow")


def results():
    rows = list(csv.DictReader((ROOT/"results/reported_metrics.csv").open()))
    f = Figure(1440, 730, "Reported unimodal and multimodal comparison: AUC, accuracy and F1")
    f.text(60,55,"Multimodal versus unimodal results",39,bold=True)
    f.text(60,108,"Reported experiment • observed missingness • values in percent",25,MUTED)
    metrics=[("AUC","auc_percent"),("Accuracy","accuracy_percent"),("F1","f1_percent")]
    labels=["Audio","Video","Text","Filter-and-Concat"]
    for col,(title,key) in enumerate(metrics):
        x=60+col*470
        f.text(x,175,title,30,bold=True)
        for i,row in enumerate(rows):
            y=230+i*95
            value=float(row[key]);is_fusion=i==3
            f.text(x,y,labels[i],21,color=TEAL if is_fusion else MUTED,bold=is_fusion)
            f.box(x,y+22,340,22,fill="#edf1f4",stroke="#edf1f4")
            f.box(x,y+22,340*value/100,22,fill=TEAL if is_fusion else "#819eaf",stroke=TEAL if is_fusion else "#819eaf")
            f.text(x+355,y+33,f"{value:.2f}",23,color=TEAL if is_fusion else INK,bold=is_fusion)
        f.text(x,620,"0",18,MUTED)
        f.text(x+340,620,"100",18,MUTED,center=True)
    f.text(60,676,"Fusion: highest recorded AUC (+0.81 pp over text); accuracy and F1 do not improve.",24,bold=True)
    f.text(60,712,"Source: supplied experiment table. No new training run; uncertainty was not reported.",20,MUTED)
    f.save("reported-results")


if __name__ == "__main__":
    workflow()
    results()
