import numpy as np
import matplotlib.pyplot as plt

# dataset
x = np.arange(0, 9)
y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
# fits line
fit = np.polyfit(x, y, 1)
fit_fn = np.poly1d(fit)
# plotting
plt.plot(x, y, 'bo', x, fit_fn(x), '--k')
plt.xlim(np.min(x) - 0.1, np.max(x) + 0.1)
plt.ylim(np.min(y) - 1, np.max(y) + 1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('linear regression')
plt.tight_layout()
plt.savefig('figures/linear_regression_example.eps')
plt.show()

