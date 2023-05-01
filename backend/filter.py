import csv
import re

raise Exception('MAKE SURE YOU SET UP THE INPUT AND OUTPUT FILEs')

with open(
    ".\data\cocktail_flavors_ingreds_popularity.csv", newline="", encoding="utf8"
) as infile, open("./data/temp.csv", "w", newline="", encoding="utf8") as outfile:
    reader = csv.reader(infile, delimiter=",")
    writer = csv.writer(outfile, delimiter=",")
    for row in reader:
        line = "$".join(row)
        line = re.sub("[0-9]*[.]*[0-9]+ oz ", "", line)
        line = re.sub("[0-9]*[.]*[0-9]+ cup[s]* ", "", line)
        line = re.sub("[0-9]*[.]*[0-9]+ ml ", "", line)
        line = re.sub("[0-9]*[.]*[0-9]+ tablespoon[s]* ", "", line)
        line = re.sub("[0-9]*[.]*[0-9]+ tbsp[s]*", "", line)
        line = re.sub("[0-9]*[.]*[0-9]+ tsp[s]*", "", line)
        line = line.split("$")
        writer.writerow(line)
