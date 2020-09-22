import sys


if __name__ == "__main__":
    files = sys.argv[1:]
    aligns = []
    for f in files:
        with open(f) as ai:
            aligns.append([set(pairs.strip().split()) for pairs in ai])
    for a in zip(*aligns):
        # a[0] for baseline and a[1] for refinement
        refine = a[0]
        f, e = {p.split("-")[0] for p in a[0]}, {p.split("-")[1] for p in a[0]}
        for pair in a[1]:
            fe = pair.split("-")
            off_diag = abs(int(fe[1]) - int(fe[0]))
            if fe[0] not in f and fe[1] not in e and off_diag <= 7:
                sys.stdout.write(pair + " ")
        for p in refine:
            sys.stdout.write(p + " ")
        sys.stdout.write("\n")
