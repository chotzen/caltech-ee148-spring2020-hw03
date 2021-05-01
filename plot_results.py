import matplotlib.pyplot as plt

tr_losses = [0.1584, 0.1034, 0.0781, 0.0541, 0.0387]
te_losses = [0.1452, 0.0939, 0.0713, 0.0524, 0.0458]
fractions = [1/16, 1/8, 1/4, 1/2, 1]

plt.loglog(fractions, tr_losses, color='red', label="Training Loss")
plt.loglog(fractions, te_losses, color='blue', label="Test Loss")
plt.legend()
plt.xlabel("Log Fraction of Dataset")
plt.ylabel("Log Loss")
plt.title("Log Loss vs. Log Fraction of Dataset")
plt.show()