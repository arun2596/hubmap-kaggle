import matplotlib.pyplot as plt

# file1 = open('SomeFile (copy).txt', 'r')
# Lines = file1.readlines()

# lr = []
# loss = []
# for line in Lines:
#     if line[:9]=="Adjusting":
#         lr.append(float(line.split("Adjusting learning rate of group 0 to ")[1][:-2]))
#     elif line[:5]=="epoch":
#         loss.append(float(line.split("train_loss: ")[1].split(',')[0]))
#     else:
#         print(line)
# print(len(lr))
# print(lr[50], lr[120])
# # print(loss)

# plt.plot(lr[:120],loss[:120])
# plt.show()


file1 = open('model/output/mit-b2-onecycleaug/log.txt', 'r')
Lines = file1.readlines()

lr = []
train_loss = []
train_metric = []
valid_loss = []
valid_metric = []

for line in Lines:
    if line.find('train_loss')>-1:
        train_loss.append(float(line.split("train_loss: ")[1].split(',')[0]))
        train_metric.append(float(line.split("dice_train_metric: ")[1]))
    elif line.find("valid_loss")>-1:
        valid_loss.append(float(line.split("valid_loss: ")[1].split(',')[0]))
        valid_metric.append(float(line.split("valid_metric: ")[1]))
print(len(train_loss))
print(len(train_metric))
print(len(valid_loss))
print(len(valid_metric))

# print(lr[50], lr[120])
# print(loss)

# plt.plot(range(len(train_loss)), train_loss)
plt.plot(range(len(train_loss)), train_metric)
# plt.plot(range(len(train_loss)), valid_loss)
plt.plot(range(len(train_loss)), valid_metric)
plt.show()

