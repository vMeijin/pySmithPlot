import csv
import numpy as np

def parseCSV(nameIn, startRow=0, endRow=1e36, steps=1, transpose=False, cSymbol='i'):
    """parses given csv file and returns an array of arrays
        nameIn - complete path
        startRow - first row to read, e.g. to skip header
        endRow - list row to read
        transpose - transposes the result (columns to rows and vice versa)
    """
    data = []
    reader = csv.reader(open("%s.csv" % nameIn, "r"))
    r = 0
    while r < startRow:  # skip first rows
        reader.next()
        r += 1

    while r < endRow:  # read rows till end
        try:
            tmp = []
            for x in reader.next():
                tmp.append(complex(x.replace(" ", "").replace(cSymbol, "j")) if cSymbol in x else float(x))
            data.append(tmp)

            for _ in range(steps):
                reader.next();

            r += steps
        except:
            endRow = 0

    if transpose:
        result = np.transpose(data)
    else:
        result = data

    return np.array(result)
