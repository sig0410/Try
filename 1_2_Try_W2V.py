import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/Moon/nlp-tutorial/tripadvisor_hotel_reviews.csv')
print(df['Rating'].unique())
print(df.isnull().sum())


# PreProcessing
import re

only = []
for i in df['Review']:
    only.append(re.sub('[^a-zA-z]', ' ', i))

# 영어 아닌건 삭제

from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

data = [word for word in only if not word in stops]

# nltk패키지를 이용해서 불용어 처리

word_sequence = ' '.join(data).split()
# 띄어쓰기 기준으로 잘라서 합쳐줌

word_list = ' '.join(data).split()
word_list = list(set(word_list))
# 중복 제거

word_dict = {w : i for i, w in enumerate(word_list)}
# 각 단어에 대해 고유한 인덱스 번호 부여

voc_size = len(word_list)
print(voc_size)
# 약 5만개의 고유한 단어리스트 생성

# 중심 단어로 주변 단어 예측
skip_grams = []

for i in range(1, len(word_sequence) - 1):

    target = word_dict[word_sequence[i]]
    # 중심 단어
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
    # 중심 단어를 기준으로 앞,뒤 단어 하나씩

    for w in context:
        skip_grams.append([target, w])


print(skip_grams[:10])
# 중심 단어와 주변 단어가 잘 매칭된것을 볼 수 있다
print(len(skip_grams))


def random_batch():
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace = False)
    # np.random.choice(replace = False) : 비복원추출
    # range(len(skip_grams))사이즈에서 batch_size만큼 비복원 추출로 인덱스 부여
    # 비복원 추출로 하는 이유는 인덱스는 고유해야함으로
    # batch_size만큼 skip_gram안에 있는 랜덤한 인덱스 부여
    #


    for i in random_index:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]]) # target 단어
        # 해당 인덱스에 해당하는 값만 1로 해주고 나머지는 0으로 해줌
        # 아마 이걸로 나중에 w2v에서 각 단어의 가중치값을 뽑아낼듯?
        random_labels.append(skip_grams[i][1]) # word 단어, +1,-1로 지정했기 때문에



    return random_inputs, random_labels


# 이부분 중요 =================
# random_index를 통해 같은 중심 단어를 두번 학습하게 된다.
# labels에 있는 인덱스가 inputs에 있는 인덱스의 자리와 같은지 비교하며 학습하는듯


# model
class W2V(nn.Module):

    def __init__(self):
        super(W2V, self).__init__()

        self.W = nn.Linear(voc_size, embedding_size, bias=False)

        self.WT = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):
        hidden_layer = self.W(X)

        output_layer = self.WT(hidden_layer)

        return output_layer

# Train

if __name__ == '__main__':
    batch_size = 2
    embedding_size = 3

    model = W2V()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):

        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, target_batch)

        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

# Tensor : float type
# LongTensor : int type

# 하드웨어 환경이 좋지 못해 좋은 결과를 얻지는 못함
''' 정리 
random_index를 통해 skip_grams에 들어있는 값을 인덱스로 불러와 학습
불러온것으로 해당 값과 인덱스 번호가 같은지 비교하며 학습을 진행하며 중심 단어에 대해 주변 단어를 예측
예측값과 실제값을 비교하며 학습 진행 
'''

