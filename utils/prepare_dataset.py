def repair_atepc_dataset(fin, fout):
    fin = open(fin, 'r', encoding='utf8')
    lines = fin.readlines()
    fin.close()

    sequences = []
    sequence = []
    for line in lines:

        if len(line) > 0 and line[0] != "\n":
            sequence.append(line.strip().split(' '))

        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            sequences.append(sequence)
            sequence = []
            continue

    def is_similar(a, b):
        count = 0
        # if (len(a) == len(b)):
        #     print(a)
        #     print(b)
        for i in a:
            for j in b:
                if i[0] == j[0]:
                    count += 2

        if count / (len(a) + len(b)) > 0.7:
            return True
        else:
            return False

    same_sequencess = []
    for i in range(len(sequences)):
        if len(same_sequencess) == 0:
            same_sequences = []
            same_sequences.append(sequences[i])
            same_sequencess.append(same_sequences)
        elif is_similar(sequences[i], same_sequencess[-1][0]):
            same_sequencess[-1].append(sequences[i])
        else:
            same_sequences = []
            same_sequences.append(sequences[i])
            same_sequencess.append(same_sequences)

    new_sequences = []
    for same_sequences in same_sequencess:
        try:
            tags = ['O'] * len(same_sequences[0])
            for sequence in same_sequences:
                for i in range(len(sequence)):
                    tags[i] = sequence[i][1] if sequence[i][1] != 'O' else tags[i]

            for sequence in same_sequences:
                new_sequence = []
                for j in range(len(sequence)):
                    t = []
                    t.append(sequence[j][0])
                    t.append(tags[j])
                    t.append(sequence[j][2])
                    new_sequence.append(t)
                new_sequences.append(new_sequence)
        except Exception as e:
            # print(e.with_traceback())
            print('1', fin.name, sequence)
            print('2', fin.name, same_sequences[0])


    tmp = []
    for sequence in new_sequences:
        if sequence not in tmp:
            tmp.append(sequence)

    fout = open(fout, 'w', encoding='utf8')

    for sequence in tmp:
        for token in sequence:
            fout.write(' '.join(token) + '\n')
        fout.write('\n')
    fout.close()


if __name__ == "__main__":
    repair_atepc_dataset('../atepc_datasets/laptop/Laptops.atepc.train.dat',
                         '../atepc_datasets/laptop/Laptops.atepc.train.dat')
    repair_atepc_dataset('../atepc_datasets/laptop/Laptops.atepc.test.dat',
                         '../atepc_datasets/laptop/Laptops.atepc.test.dat')

    repair_atepc_dataset('../atepc_datasets/restaurant/Restaurants.atepc.train.dat',
                         '../atepc_datasets/restaurant/Restaurants.atepc.train.dat')
    repair_atepc_dataset('../atepc_datasets/restaurant/Restaurants.atepc.test.dat',
                         '../atepc_datasets/restaurant/Restaurants.atepc.test.dat')

    repair_atepc_dataset('../atepc_datasets/twitter/twitter.atepc.train.dat',
                         '../atepc_datasets/twitter/twitter.atepc.train.dat')
    repair_atepc_dataset('../atepc_datasets/twitter/twitter.atepc.test.dat',
                         '../atepc_datasets/twitter/twitter.atepc.test.dat')
