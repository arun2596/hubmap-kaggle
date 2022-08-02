import matplotlib.pyplot as plt

file1 = open('SomeFile (copy).txt', 'r')
Lines = file1.readlines()

lr = []
loss = []
for line in Lines:
    if line[:9]=="Adjusting":
        lr.append(float(line.split("Adjusting learning rate of group 0 to ")[1][:-2]))
    elif line[:5]=="epoch":
        loss.append(float(line.split("train_loss: ")[1].split(',')[0]))
    else:
        print(line)
print(len(lr))
print(lr[50], lr[120])
# print(loss)

plt.plot(lr[:120],loss[:120])
plt.show()