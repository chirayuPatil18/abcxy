import sys

filename = sys.argv[1]

with open(filename, 'r') as file:
    for line in file:
        line = line.lower()

        # ---------- WORD COUNT ----------
        words = line.split()

        for word in words:
            print(f"{word}\t1")

        # ---------- CHARACTER COUNT ----------
        for ch in line:
            if ch != ' ' and ch != '\n':
                print(f"{ch}\t1")
        

# python dc3mapper.py sample.txt | sort | python dc3reducer.py