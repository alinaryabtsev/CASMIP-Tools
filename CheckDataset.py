from glob import glob
import os
import json

NO_CONTRAST_PHASE = ["NO_CONTRAST", "NCCT", "NON_CONTRAST"]
ARTERIAL_PHASE = ["ARTERIAL", "PORTAL"]  # PORTAL is a mistake in Ein Karem systems, it refers for arterial phase
VENOUS_PHASE = ["VENOUS"]
LIVER_AND_ABDOMEN = ["LIVER", "ABDOMEN"]

PHASE = "ImageComments"
BODY_PART = "BodyPartExamined"

dirs = glob("/cs/casmip/alina.ryabtsev/PrimaryLesionsProject/Dataset/CT/*/*/*")


for d in dirs:
    print(f"In case {d}:")
    jsons = glob(os.path.join(d, "*.json"))
    for json_file in jsons:
        with open(json_file) as file:
            data = json.load(file)

        try:
            phase = data[PHASE]
            body_part = data[BODY_PART]
            if body_part in LIVER_AND_ABDOMEN and phase in NO_CONTRAST_PHASE + ARTERIAL_PHASE + VENOUS_PHASE:
                print(f"    - {phase} phase is file: {json_file}")
        except KeyError:
            continue
    print()


