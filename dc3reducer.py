import sys

words = {}
chars = {}

for line in sys.stdin:
    key, value = line.strip().split('\t')
    value = int(value)

    if len(key) == 1:
        chars[key] = chars.get(key, 0) + value
    else:
        words[key] = words.get(key, 0) + value

print("WORD COUNT")
for k in sorted(words):
    print(f"{k}\t{words[k]}")

print("\nCHARACTER COUNT")
for k in sorted(chars):
    print(f"{k}\t{chars[k]}")