import csv
import re

raise Exception('MAKE SURE YOU SET UP AN OUTPUT FILE')

with open(
    ".\data\cocktail_flavors_ingreds_combined.csv", newline="", encoding="utf8"
) as infile, open("./data/test.csv", "w", newline="", encoding="utf8") as outfile:  # change 'test.csv' to output file
    reader = csv.reader(infile, delimiter=",")
    writer = csv.writer(outfile, delimiter=",")
    for row in reader:
        line = "$".join(row)
        line = re.sub("[0-9]*[.]*[0-9]+ oz ", "", line)
        line = re.sub("[0-9]*[.]*[0-9]+ cup[s]* ", "", line)
        line = re.sub("[0-9]*[.]*[0-9]+ ml ", "", line)
        line = line.split("$")
        writer.writerow(line)
