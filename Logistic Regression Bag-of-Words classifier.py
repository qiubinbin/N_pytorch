"""逻辑回归词袋分类器"""
import torch
import torch.nn.functional as F

train_data = [('me gusta comer en la cafeteria'.split(), 'SPANISH'),
              ('Give it to me'.split(), 'ENGLISH'),
              ('No creo que sea una buena idea'.split(), 'SPANISH'),
              ('No it is not a good idea to get lost at sea'.split(), 'ENGLISH')]
test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]
word_to_index = {}
for sentence, _ in train_data + test_data:
    for word in sentence:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
size_vocabulary = len(word_to_index)


class Lrbag(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, sentence):
        return F.log_softmax(self.fc(sentence))  # 如果不想手动添加柔性最大值层，可以用CrossEntropyLoss代替,F中的函数不具备可学习参数


model = Lrbag(size_vocabulary, 2)
target = {'SPANISH': 0, 'ENGLISH': 1}
epochs = 20


def transform(sample, sample_dict):
    input = torch.zeros(len(sample_dict))
    for word in sample:
        input[word_to_index[word]] += 1
    return input.view(1, -1)  # 特别注意输出第一维应该是小批量尺寸(N*d1*d2...)


# for data, label in train_data:
#     print(make_input(data))
# for i in range(20):
loss_fn = torch.nn.NLLLoss()  # 负对数似然柔性最大值代价函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(epochs):
    for data, label in train_data:
        data_input = transform(data, word_to_index)
        label_input = torch.LongTensor([target[label]])
        output = model(data_input)
        loss = loss_fn(output, label_input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for data, label in test_data:
    data = transform(data, word_to_index)
    output = model(data)
    print('Pred: ', torch.argmax(output), 'Target:', label)
