import matplotlib.pyplot as plt

data = open("log.txt", "r").readlines()

data = [line.split(":")[1].strip() for line in data if line.startswith("Accuracy")]

#print(data)

cnn_data = [float(data[i]) for i in range(0, len(data), 2)]
nn_data = [float(data[i]) for i in range(1, len(data), 2)]

print(cnn_data)
# print(nn_data)

plt.ylim(0.5, 1)

plt.plot(range(len(cnn_data)),  cnn_data)
plt.plot(range(len(nn_data)), nn_data)
plt.legend(["CNN", "NN"])
plt.xlabel("Training instance")
plt.ylabel("Accuracy")
plt.show()