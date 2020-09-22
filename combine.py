import sys
from collections import defaultdict

if __name__ == "__main__":
    files = sys.argv[1:]
    aligns = []
    for f in files:
        with open(f) as ai:
            aligns.append([set(pairs.strip().split()) for pairs in ai])
    for a in zip(*aligns):
        acount = defaultdict(int)
        for ps in a:
            for p in ps:
                acount[p] += 1
#         pairs = set.intersection(*a)
        for p, count in acount.items():
            if count > len(a) * 0.5:
                sys.stdout.write(p + " ")
        sys.stdout.write("\n")