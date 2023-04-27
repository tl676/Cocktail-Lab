import csv
import re

raise Exception('MAKE SURE YOU SET UP THE INPUT AND OUTPUT FILEs')

with open(
    ".\data\________.csv", newline="", encoding="utf8"
) as infile, open("./data/_______.csv", "w", newline="", encoding="utf8") as outfile:
    reader = csv.reader(infile, delimiter=",")
    writer = csv.writer(outfile, delimiter=",")
    for row in reader:
        line = "$".join(row)
        line = re.sub("[0-9]*[.]*[0-9]+ oz ", "", line)
        line = re.sub("[0-9]*[.]*[0-9]+ cup[s]* ", "", line)
        line = re.sub("[0-9]*[.]*[0-9]+ ml ", "", line)
        line = line.split("$")
        writer.writerow(line)
