from graph_construction.vlm_annotation_parse.qwen35vl_meccano_parse import parse_annotate_action

test = (
    "The camera wearer is holding their hands together over a wooden table "
    "with scattered red and gray LEGO-like pieces, a QR code sheet, a booklet "
    "with diagrams, a black cylindrical object, a gray perforated bar, "
    "a red angled perforated bar, a red 4 perforated junction bar, "
    "a red perforated junction bar, a red perforated bar, a red rod, "
    "a gray angled perforated bar, a white angled perforated bar, "
    "a red handlebar, a red wrench, a gray"
)

result = parse_annotate_action(test)
subject, verb, verb_type, direct_object, all_objects_map = (
    result[0], result[1], result[2], result[3], result[4]
)

print("Subject:", subject)
print("Verb:", verb)
print("Direct object:", direct_object)
print()
print("All objects:")
for k, v in all_objects_map.items():
    print(f"  {k!r}: base={v['base_object']!r}, attrs={v['attributes']}")
