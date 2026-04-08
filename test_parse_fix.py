import sys, os
os.environ["PYTHONNOUSERSITE"] = "1"
sys.path.insert(0, '/home/s3758869/egocentric_video_graph_framework_ar')
from graph_construction.vlm_annotation_parse.qwen35vl_meccano_parse import parse_annotate_action

def show(label, sentence):
    print(f'=== {label} ===')
    result = parse_annotate_action(sentence)
    print('direct_object:', result[3])
    print('all_objects:')
    for k, v in result[4].items():
        print(f'  {repr(k)}: {v}')
    print('aux_verbs:', result[12])
    print('object_aux_verb:', result[13])
    print()

show("TEST 1: SHORT",
     "The camera wearer is assembling a model using red and gray perforated bars, red angled perforated bars")

show("TEST 2: LONG",
     "The camera wearer is assembling a model using red and gray perforated bars, red angled perforated bars, red perforated junction bars, a gray angled perforated bar, a red 4 perforated junction bar, a gray perforated bar, a red rod, a red handlebar, a red wrench, a red screwdriver, a red tire, a gray rim, a red roller, a red nut, a red washer, a red screw, a red wheel axle, a red partial model")

show("TEST 3: BOOKLET",
     "The camera wearer is assembling a model using red and gray plastic pieces including a red perforated bar, red angled perforated bar, gray angled perforated bar, white angled perforated bar, wrench, and gray perforated bar, with a booklet containing assembly instructions placed on a wooden table alongside scattered components and a QR code.")

# Debug: show POS/tag/dep for every token in TEST 3
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The camera wearer is assembling a model using red and gray plastic pieces including a red perforated bar, red angled perforated bar, gray angled perforated bar, white angled perforated bar, wrench, and gray perforated bar, with a booklet containing assembly instructions placed on a wooden table alongside scattered components and a QR code.")
print("=== DEBUG: TOKEN TABLE ===")
for tok in doc:
    marker = " <<<" if tok.lemma_ == "angle" else ""
    print(f"  {tok.i:3d} {tok.text:20s} pos={tok.pos_:6s} tag={tok.tag_:5s} dep={tok.dep_:10s} head={tok.head.text:20s} head.pos={tok.head.pos_:6s}{marker}")
