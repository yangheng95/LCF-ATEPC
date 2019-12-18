import os
import copy

def assemble_aspects(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('$ t $','$T$').strip()

    def is_similar(s1, s2):
        count = 0.0
        for token in s1.split(' '):
            if token in s2:
                count += 1
        # if count / len(s1.split(' ')) >= 0.7 and abs(len(s1.split(' '))-len(s2.split(' '))<5):
        if count / len(s1.split(' ')) >= 0.7 and count / len(s2.split(' ')) >= 0.7:
            return True
        else:
            return False

    def unify_same_samples(same_samples):
        text = same_samples[0][0].replace('$T$', same_samples[0][1])
        polarities = [-1]*len(text.split())
        tags=['O']*len(text.split())
        samples = []
        for sample in same_samples:
            polarities_tmp = copy.deepcopy(polarities)

            try:
                asp_begin = (sample[0].split().index('$T$'))
                asp_end = sample[0].split().index('$T$')+len(sample[1].split())
                for i in range(asp_begin, asp_end):
                    polarities_tmp[i] = int(sample[2])+1
                    if i - sample[0].split().index('$T$')<1:
                        tags[i] = 'B-ASP'
                    else:
                        tags[i] = 'I-ASP'
            except:
                pass
            samples.append([text, tags, polarities_tmp])
        return samples

    samples = []
    aspects_in_one_sentence = []
    for i in range(0, len(lines), 3):

        # aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])

        if len(aspects_in_one_sentence) == 0:
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
            continue
        if is_similar(aspects_in_one_sentence[-1][0], lines[i]):
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
        else:
            samples.extend(unify_same_samples(aspects_in_one_sentence))
            aspects_in_one_sentence = []
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])

     # for sample in samples:
    #     aspects = []
    #     for sentence in sentences:
    #         aspects.append((sentence[1], sentence[0].split().index('$T$')))
    #
    #     for sentence in sentences:
    #         # 将无关aspect屏蔽，其中有关的的aspect被写作$T$
    #         for aspect in aspects:
    #             sentence[0]=sentence[0].replace(aspect[0],'[MASK]')
    #             # sentence[0] = sentence[0].replace(aspect[0], '')  # .replace('$T$','')
    #             tokens=sentence[0].split()
    #             try:
    #                 index = tokens.index('[MASK]')
    #             except:
    #                 continue
    #             # for i in range(len(tokens)):
    #             #         if abs(i - index) <= 3:
    #             #            tokens[i]='$MASK$'
    #             # sentence[0]=' '.join(tokens)
    #
    #     for sentence in sentences:
    #         sentence[0].replace('$MASK$','[MASK]')

    return samples


def split_aspects(sentence):
    single_aspect_with_contex = []

    aspect_num = len(sentence[1].split("|"))
    aspects = sentence[1].split("|")
    polarity = sentence[2].split("|")
    pre_position = 0
    aspect_contex = sentence[0]
    for i in range(aspect_num):
        aspect_contex = aspect_contex.replace("$A$", aspects[i], 1)
        single_aspect_with_contex.append(
            (aspect_contex[pre_position:aspect_contex.find("$A$")], aspects[i], polarity[i]))
        pre_position = aspect_contex.find(aspects[i]) + len(aspects[i]) + 1

    return single_aspect_with_contex


# 将数据集中的aspect切割出来
def refactor_dataset(fname, dist_fname):
    lines = []
    samples = assemble_aspects(fname)

    for sample in samples:
        for token_index in range(len(sample[1])):
            token, label, polarty = sample[0].split()[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarty))

        lines.append('\n')
    # 写之前，先检验文件是否存在，存在就删掉
    if os.path.exists(dist_fname):
        os.remove(dist_fname)
    fout = open(dist_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line+'\n').replace('\n\n', '\n'))
    fout.close()

# 将数据集中的aspect切割出来
def refactor_chinese_dataset(fname, train_fname,test_fname):
    lines = []
    samples = assemble_aspects(fname)
    positive = 0
    negative = 0
    sum = 0
    # refactor testset
    for sample in samples[:int(len(samples)/5)]:
        for token_index in range(len(sample[1])):
            token, label, polarty = sample[0].split()[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarty))
        lines.append('\n')
        if 1 in sample[2]:
            positive+=1
        else:negative+=1
        sum+=1
    print(train_fname+f"sum={sum} positive={positive} negative={negative}")
    if os.path.exists(test_fname):
        os.remove(test_fname)
    fout = open(test_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line+'\n').replace('\n\n', '\n'))
    fout.close()

    positive = 0
    negative = 0
    sum = 0
    # refactor trainset
    for sample in samples[int(len(samples)/5):]:
        for token_index in range(len(sample[1])):
            tokens = sample[0].split()
            token, label, polarty = sample[0].split()[token_index], sample[1][token_index], sample[2][token_index]
            lines.append(token + " " + label + " " + str(polarty))
        lines.append('\n')
        if 1 in sample[2]:
            positive+=1
        else:negative+=1
        sum+=1
    print(train_fname+f"sum={sum} positive={positive} negative={negative}")
    if os.path.exists(train_fname):
        os.remove(train_fname)
    fout = open(train_fname, 'w', encoding='utf8')
    for line in lines:
        fout.writelines((line + '\n').replace('\n\n', '\n'))
    fout.close()


if __name__ == "__main__":

    # chinese datasets
    refactor_chinese_dataset(
        r"../chinese_atepc_datasets/camera_output.txt",
        r"../chinese_atepc_datasets/camera.atepc.train.dat",
        r"../chinese_atepc_datasets/camera.atepc.test.dat",
    )
    refactor_chinese_dataset(
        r"../chinese_atepc_datasets/car_output.txt",
        r"../chinese_atepc_datasets/car.atepc.train.dat",
        r"../chinese_atepc_datasets/car.atepc.test.dat",
    )
    refactor_chinese_dataset(
        r"../chinese_atepc_datasets/notebook_output.txt",
        r"../chinese_atepc_datasets/notebook.atepc.train.dat",
        r"../chinese_atepc_datasets/notebook.atepc.test.dat",
    )
    refactor_chinese_dataset(
        r"../chinese_atepc_datasets/phone_output.txt",
        r"../chinese_atepc_datasets/phone.atepc.train.dat",
        r"../chinese_atepc_datasets/phone.atepc.test.dat",
    )



    # #  推特数据集
    # refactor_dataset(
    #     r"datasets/acl-14-short-data/train.raw",
    #     r"atepc_datasets/twitter/twitter.atepc.train.dat",
    # )

    # # 笔记本数据集
    # refactor_dataset(
    #     r"datasets/semeval14/Laptops_Test_Gold.xml.seg",
    #     r"atepc_datasets/laptop/Laptops.atepc.laptop_test.dat",
    # )
    # refactor_dataset(
    #     r"datasets/semeval14/Laptops_Train.xml.seg",
    #     r"atepc_datasets/laptop/Laptops.atepc.train.dat",
    # )
    # # 餐厅数据集
    # refactor_dataset(
    #     r"datasets/semeval14/Restaurants_Train.xml.seg",
    #     r"atepc_datasets/restaurant/Restaurants.atepc.train.dat",
    # )
    # refactor_dataset(
    #     r"datasets/semeval14/Restaurants_Test_Gold.xml.seg",
    #     r"atepc_datasets/restaurant/Restaurants.atepc.laptop_test.dat",
    # )
    # #  推特数据集
    # refactor_dataset(
    #     r"datasets/acl-14-short-data/train.raw",
    #     r"atepc_datasets/twitter/twitter.atepc.train.dat",
    # )
    # refactor_dataset(
    #     r"datasets/acl-14-short-data/laptop_test.raw",
    #     r"atepc_datasets/twitter/twitter.atepc.laptop_test.dat",
    # )
