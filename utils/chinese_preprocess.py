import os
import copy

def preprocess_chinese(path):
        pass

def read_chinese(path):
        output_path = path + '_output.txt'
        content = path + '_sentence.txt'
        aspect = path + '_target.txt'
        polarity = path + '_label.txt'
        fin = open(content, 'r', encoding='utf-8', newline='\n', errors='ignore')
        reviews = fin.readlines()
        fin.close()
        for i in range(len(reviews)):
            reviews[i] = reviews[i].strip()

        fin = open(aspect, 'r', encoding='utf-8', newline='\n', errors='ignore')
        aspects = fin.readlines()
        fin.close()
        for i in range(len(aspects)):
            aspects[i] = aspects[i].strip()

        fin = open(polarity, 'r', encoding='utf-8', newline='\n', errors='ignore')
        polarities = fin.readlines()
        fin.close()
        for i in range(len(polarities)):
            polarities[i] = polarities[i].strip()

        from pytorch_transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert_for_global_context-base-chinese', do_lower_case=True)

        with open(output_path, 'w', encoding='utf-8', newline='\n', errors='ignore') as f_out:
            print(len(reviews))
            print(len(aspects))
            print(len(polarities))

            for i in range(len(reviews)):

                if aspects[i] is '0':
                    aspects[i] = reviews[i]
                if aspects[i].replace(' ','') not in reviews[i]:
                    print(aspects[i].replace(' ',''))
                    continue

                reviews[i]=reviews[i].replace(aspects[i].replace(' ',''),' $T$ ')

                f_out.write(' '.join(tokenizer.tokenize(reviews[i])) + '\n')
                f_out.write(' '.join(tokenizer.tokenize(aspects[i].replace(' ',''))) + '\n')
                if polarities[i].strip() is '0':
                    f_out.write('1' + '\n')
                else:
                    f_out.write('-1' + '\n')

            f_out.close()


if __name__ == "__main__":

   car_path = r'C:\Users\chuan\Desktop\Chinese review datasets\car'
   phone_path = r'C:\Users\chuan\Desktop\Chinese review datasets\phone'
   camera_path = r'C:\Users\chuan\Desktop\Chinese review datasets\camera'
   notebook_path = r'C:\Users\chuan\Desktop\Chinese review datasets\notebook'

   read_chinese(car_path)
   read_chinese(phone_path)
   read_chinese(camera_path)
   read_chinese(notebook_path)