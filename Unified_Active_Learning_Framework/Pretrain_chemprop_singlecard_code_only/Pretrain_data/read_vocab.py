# data = []
#
# with open('simple_vocab.txt', 'r') as file:
#     for line in file:
#         row = line.strip().split(',')
#         row = (row[0], int(row[1].strip()))
#         data.append(row)
#
# print(data)
#
# with open('complex_vocab_v2.txt', 'r') as file:
#     for line in file:
#         row = line.strip().split(',')
#         row = (row[0], int(row[1].strip()))
#         data.append(row)
#
# print(data)

import csv

# Open the file
with open('complex_vocab_v3.txt', 'r') as file:
    reader = csv.reader(file)

    # Create a list of tuples
    rows = []
    for row in reader:
        # Convert the digitals into integers
        new_row = []
        for element in row:
            if element.strip() == '-1':
                new_row.append(-1)
            if element.strip() == '-2':
                new_row.append(-2)
            elif element.strip().isdigit():
                new_row.append(int(element.strip()))
            else:
                new_row.append(element.strip())
        rows.append(tuple(new_row))

print(rows)