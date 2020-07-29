import csv

def transform_to_tsv(size):
    train_file = open("roberta/dataset/labeledTrainData_full.tsv", "w+")
    tsv_writer_train = csv.writer(train_file, delimiter='\t')
    test_file = open("roberta/dataset/testData_full.tsv", "w+")
    tsv_writer_test = csv.writer(test_file, delimiter='\t')
    tsv_writer_test.writerow(['tweet'])
    tsv_writer_train.writerow(['tweet', 'sentiment'])
    neg = open("src/data/train_neg_full.txt", "r")
    pos = open("src/data/train_pos_full.txt", "r")
    test = open("src/data/test_data.txt", "r")
    for i, line in enumerate(pos.readlines()):
        if i < size:
            tsv_writer_train.writerow([line.replace("\n",""), "1"])
    for i, line in enumerate(neg.readlines()):
        if i < size:
            tsv_writer_train.writerow([line.replace("\n",""), "0"])
    for line in test.readlines():
        tsv_writer_test.writerow([line.split(",")[1].replace("\n", "")])
    train_file.close()
    test_file.close()
    neg.close()
    pos.close()
    test.close()


def transform_to_tsv_proc(size):
    train_file = open("roberta/dataset/labeledTrainData_proc_full.tsv", "w+")
    tsv_writer_train = csv.writer(train_file, delimiter='\t')
    test_file = open("roberta/dataset/testData_proc.tsv", "w+")
    tsv_writer_test = csv.writer(test_file, delimiter='\t')
    tsv_writer_test.writerow(['tweet'])
    tsv_writer_train.writerow(['tweet', 'sentiment'])
    neg = open("src/data/preproc_neg_full.txt", "r")
    pos = open("src/data/preproc_pos_full.txt", "r")
    test = open("src/data/preproc_test.txt", "r")
    last_line = ""
    for i, line in enumerate(pos.readlines()):
        if i < size:
            if last_line == line:
                pass
            else:
                tsv_writer_train.writerow([line.replace("\n",""), "1"])
            last_line = line
    for i, line in enumerate(neg.readlines()):
        if i < size:
            if last_line == line:
                pass
            else:
                tsv_writer_train.writerow([line.replace("\n",""), "0"])
            last_line = line
    for line in test.readlines():
        tsv_writer_test.writerow([line.replace("\n", "")])
    train_file.close()
    test_file.close()
    neg.close()
    pos.close()
    test.close()



if __name__ == '__main__':
    transform_to_tsv_proc(2500000)