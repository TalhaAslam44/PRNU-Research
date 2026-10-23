"""Fill the graduate-workshop template (fonts, sizes, layout untouched): text, two figures, a caption, speaker notes."""
import re, shutil, sys
from pathlib import Path
from defusedxml import minidom
from PIL import Image

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
FIG = Path("/home/talhaaslam/PRNU-Research/report/figures")
EMU = 914400
shutil.rmtree(OUT, ignore_errors=True); shutil.copytree(SRC, OUT)
shutil.copy(FIG / "poster_framework.png", OUT / "ppt/media/image2.png")
shutil.copy(FIG / "poster_results.png", OUT / "ppt/media/image3.png")

BODY = {
 (1, "TextBox 13"): [("Sensor fingerprint (PRNU):", " a unique noise pattern that links a video to its camera"),
                     ("Threat:", " attackers remove it, forge another camera's PRNU, or inject it into AI-generated video"),
                     ("Gap:", " the conventional PCE test cannot tell organic from forged sensor noise")],
 (2, "TextBox 13"): [("Analyse", " PRNU in genuine and attacked video"),
                     ("Identify", " features of organic vs. synthetic noise"),
                     ("Develop", " a hybrid PRNU + ML/DL framework"),
                     ("Evaluate", " robustness to re-sharing and attacks"),
                     ("Improve", " on conventional PRNU verification")],
 (3, "TextBox 13"): [("Data:", " 124 cameras, 8 AI generators, 27,583 clips"),
                     ("Attacks:", " PRNU removal / injection, denoising"),
                     ("Conditions:", " WhatsApp / YouTube-like re-sharing"),
                     ("Evidence:", " PRNU, Noiseprint++, temporal noise, ViT"),
                     ("Models:", " SVM, XGBoost, GRU, Transformer, stacking"),
                     ("Protocol:", " 5-fold CV grouped by camera model")],
 (6, "TextBox 1"): [("RQ1:", " Noiseprint++ and temporal noise consistency are most discriminative"),
                    ("RQ2:", " re-sharing and stabilisation weaken PRNU most"),
                    ("RQ3:", " HVPF AUC 0.941 vs. 0.549 (PCE); injected AI accepted 0–1% vs. 98%"),
                    ("Next:", " re-shared video and adaptive attackers")],
 (7, "TextBox 1"): [("", "Lukáš et al., IEEE TIFS, 2006 (PRNU)"),
                    ("", "Khan et al., ISC 2025 (PRNU transfer attacks)"),
                    ("", "Elharrouss et al., PDC-ViT, Neural Comput. Appl., 2025"),
                    ("", "Bao et al., Sensors, 2024 (video anti-forensics)"),
                    ("", "Shullani et al., EURASIP JIS, 2017 (VISION)"),
                    ("", "Guillaro et al., CVPR 2023 (Noiseprint++)")],
}
PICS = {4: ("Picture 11", "image2.png", 3.42, 1.25, 9.45, 3.75), 5: ("Picture 12", "image3.png", 3.42, 1.18, 9.45, 4.2)}
CAPTION = (4, "Figure: HVPF fuses PRNU evidence, noise statistics and a Vision Transformer by stacking")
NOTES = {
 1: "Every camera sensor has tiny manufacturing differences that leave a fixed noise pattern, the PRNU, in every video. "
    "Analysts correlate a video's noise with a camera's fingerprint (the PCE test) to verify where it came from. "
    "An attacker can subtract that fingerprint, add another camera's, or add a real camera's fingerprint to an AI-generated "
    "video so it passes as genuine. The PCE test only checks whether the fingerprint is present, so it cannot tell genuine "
    "sensor noise from forged noise. That is the gap this thesis addresses.",
 2: "Five objectives. First, analyse how the PRNU fingerprint behaves in genuine videos and in videos where it was removed "
    "or forged. Second, identify the features that separate genuine sensor noise from synthetic or manipulated noise. Third, "
    "develop a hybrid framework that combines PRNU analysis with machine learning and deep learning. Fourth, evaluate how "
    "robust it is to compression, social-media re-sharing and anti-forensic attacks. Fifth, improve on the conventional PRNU "
    "test. These objectives lead to three research questions, which I answer on the conclusion slide.",
 3: "Data: three real-camera datasets (VISION, Video-ACID, FloreView; 124 cameras) and eight AI video generators, 27,583 clips. "
    "The manipulated class is created by attacking real videos: PRNU removal, PRNU injection and denoising, plus WhatsApp- and "
    "YouTube-like copies; AI clips also get a real camera's PRNU injected. From the first five I-frames we compute PRNU scores, "
    "Noiseprint++ statistics, temporal noise consistency and noise residuals for a Vision Transformer. Evaluation is 5-fold "
    "cross-validation grouped by camera model, so test cameras are never seen in training; FloreView phones and four generators "
    "are kept completely unseen.",
 4: "Each clip, with the camera it claims to come from, is reduced to its first five I-frames. Three kinds of evidence are "
    "computed: PRNU match scores against the claimed camera and the best other camera, noise statistics such as Noiseprint++ "
    "and temporal consistency, and SRM noise residuals. Classical and sequence models use the features, and a Vision "
    "Transformer looks at the residuals directly. A stacking model, trained only on validation cameras, combines them into "
    "organic, manipulated or synthetic. The conventional PCE rule is the baseline.",
 5: "Figure 1: on camera models never seen in training, HVPF reaches an AUC of 0.941 for genuine versus not genuine, against "
    "0.549 for the conventional PCE rule; its three-class macro F1 is 0.851. Figure 2: AI videos with an injected camera "
    "fingerprint pass the PCE rule 98% of the time for known generators and 96% for unseen ones, but HVPF accepts only 0% "
    "and 1%. Khan et al. report 85.5% for the same attack on images.",
 6: "RQ1: removing Noiseprint++ statistics costs the most accuracy, followed by temporal noise consistency and PRNU. RQ2: "
    "platform re-compression and electronic stabilisation weaken PRNU the most; for example, WhatsApp copies lower PRNU camera "
    "identification from 100% to 61%. RQ3: yes, the hybrid framework clearly beats the conventional test and stops PRNU "
    "injection. Next steps: better decisions for heavily re-shared videos and an attacker who knows the detector.",
 7: "Lukáš et al. introduced PRNU camera identification. Khan et al. showed PRNU transfer attacks on images. PDC-ViT "
    "(Elharrouss et al.) is the closest Transformer baseline. Bao et al. use inter-frame noise for video forgery detection. "
    "VISION is the main camera dataset, and Noiseprint++ comes from TruFor (Guillaro et al.).",
}
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"

def write(doc, path, raw):
    path.write_text(raw[: raw.index("?>") + 2] + doc.documentElement.toxml(), encoding="utf-8")

def set_run(run, text, bold=False):
    if bold:
        run.getElementsByTagName("a:rPr")[0].setAttribute("b", "1")
    run.getElementsByTagName("a:t")[0].firstChild.data = text
    if text != text.strip():
        run.getElementsByTagName("a:t")[0].setAttribute("xml:space", "preserve")

for n in range(1, 8):
    path = OUT / f"ppt/slides/slide{n}.xml"
    raw = path.read_text(encoding="utf-8")
    raw = raw.replace("Title:  Malicious PDF Detections System  ", "Title:  Robust Video Source Provenance")
    raw = raw.replace("Ahmed Abdullah 24i-1234 MS(DS)\t\t\tSupervisor: Dr. Name",
                      "Talha Aslam 24i-8067 MS(DS)  |  Supervisor: Dr. Hassan Mujtaba  |  Co-Supervisor: Dr. Muhammad Ishtiaq")
    doc = minidom.parseString(raw)
    for sp in doc.getElementsByTagName("p:sp"):
        name = sp.getElementsByTagName("p:cNvPr")[0].getAttribute("name")
        if (n, name) not in BODY:
            continue
        tx = sp.getElementsByTagName("p:txBody")[0]
        paras = [p for p in tx.childNodes if p.nodeName == "a:p"]
        proto = next(p for p in paras if p.getElementsByTagName("a:r"))
        for p in paras:
            tx.removeChild(p)
        for label, text in BODY[(n, name)]:
            p = proto.cloneNode(True)
            runs = p.getElementsByTagName("a:r")
            for r in runs[1:]:
                p.removeChild(r)
            body_run = runs[0]
            if label:
                lab = body_run.cloneNode(True)
                set_run(lab, label, bold=True)
                p.insertBefore(lab, body_run)
            set_run(body_run, text)
            tx.appendChild(p)
    if n in PICS:
        pname, media, x0, y0, wmax, hmax = PICS[n]
        im = Image.open(OUT / "ppt/media" / media); ar = im.size[0] / im.size[1]
        w, h = (wmax, wmax / ar) if wmax / ar <= hmax else (hmax * ar, hmax)
        x, y = x0 + (wmax - w) / 2, y0 + (hmax - h) / 2
        for pic in doc.getElementsByTagName("p:pic"):
            if pic.getElementsByTagName("p:cNvPr")[0].getAttribute("name") == pname:
                off = pic.getElementsByTagName("a:off")[0]
                ext = [e for e in pic.getElementsByTagName("a:ext") if e.hasAttribute("cx")][0]
                off.setAttribute("x", str(int(x * EMU))); off.setAttribute("y", str(int(y * EMU)))
                ext.setAttribute("cx", str(int(w * EMU))); ext.setAttribute("cy", str(int(h * EMU)))
                bottom = y + h
    if n == CAPTION[0]:
        next_id = max(int(v) for v in re.findall(r'<p:cNvPr id="(\d+)"', raw)) + 1
        cap = minidom.parseString(
            f'<p:sp xmlns:p="{NS_P}" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<p:nvSpPr><p:cNvPr id="{next_id}" name="Figure caption"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{int(3.42 * EMU)}" y="{int((bottom + 0.06) * EMU)}"/><a:ext cx="{int(9.45 * EMU)}" cy="{int(0.36 * EMU)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" rtlCol="0"><a:spAutoFit/></a:bodyPr><a:lstStyle/>'
            f'<a:p><a:pPr algn="ctr"/><a:r><a:rPr lang="en-US" sz="1400" i="1" dirty="0"/><a:t>{CAPTION[1]}</a:t></a:r></a:p>'
            f'</p:txBody></p:sp>').documentElement
        tree = doc.getElementsByTagName("p:spTree")[0]
        tree.appendChild(doc.importNode(cap, True))
    write(doc, path, raw)

# speaker notes: one notes slide per slide, linked to the template's notes master
ct = (OUT / "[Content_Types].xml").read_text()
for n, text in NOTES.items():
    notes = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
             f'<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
             f'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="{NS_P}">'
             f'<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
             f'<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
             f'<p:sp><p:nvSpPr><p:cNvPr id="2" name="Slide Image Placeholder 1"/><p:cNvSpPr><a:spLocks noGrp="1" noRot="1" noChangeAspect="1"/></p:cNvSpPr>'
             f'<p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr><p:spPr/></p:sp>'
             f'<p:sp><p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
             f'<p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>'
             f'<a:p><a:r><a:rPr lang="en-US" dirty="0"/><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>'
             f'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>')
    (OUT / "ppt/notesSlides/_rels").mkdir(parents=True, exist_ok=True)
    (OUT / f"ppt/notesSlides/notesSlide{n}.xml").write_text(notes, encoding="utf-8")
    (OUT / f"ppt/notesSlides/_rels/notesSlide{n}.xml.rels").write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>'
        f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/slide{n}.xml"/></Relationships>',
        encoding="utf-8")
    rels = OUT / f"ppt/slides/_rels/slide{n}.xml.rels"
    r = rels.read_text()
    rid = "rId" + str(max(int(v) for v in re.findall(r'Id="rId(\d+)"', r)) + 1)
    r = r.replace("</Relationships>", f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" '
                  f'Target="../notesSlides/notesSlide{n}.xml"/></Relationships>')
    rels.write_text(r)
    ct = ct.replace("</Types>", f'<Override PartName="/ppt/notesSlides/notesSlide{n}.xml" '
                    'ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/></Types>')
(OUT / "[Content_Types].xml").write_text(ct)
print("built", OUT)
