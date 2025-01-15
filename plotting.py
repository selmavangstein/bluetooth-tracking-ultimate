import numpy as np
import matplotlib.pyplot as plt

# Define the function f(n)
def f1(n):
    # Avoid division by zero or invalid square roots by masking invalid domains
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.sqrt((1 - 3/n) / (n - 5 + 6/n))
        term2 = np.sqrt(((1 - 3/n) * 27) / (n * (n - n / (n - 2))))
        numerator = -term1 + term2
        denominator = 1 - np.sqrt(27 * n) / (n**2)
        result = numerator / denominator
        # Handle invalid values for square roots and divisions
        result = np.where(np.isnan(result) | np.isinf(result), np.nan, result)
    return result


def f2(n):
    # Avoid division by zero or invalid square roots by masking invalid domains
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.sqrt((1 - 3/n) / (n - 5 + 6/n))
        term2 = np.sqrt(((1 - 3/n) * 27) / (n * (n - n / (n - 2))))
        numerator = -term1 - term2
        denominator = 1 + np.sqrt(27 * n) / (n**2)
        result = numerator / denominator
        # Handle invalid values for square roots and divisions
        result = np.where(np.isnan(result) | np.isinf(result), np.nan, result)
    return result

# Define the range for n
n = np.linspace(0, 100, 10000)  # Start from 6 to avoid division by zero and invalid domains
f1_values = f1(n)
f2_values = f2(n)

# Compute arcsin(f(n)) where valid
arcsin_values1 = np.arcsin(f1_values[np.isfinite(f1_values) & (np.abs(f1_values) <= 1)])
arcsin_values2 = np.arcsin(f2_values[np.isfinite(f2_values) & (np.abs(f2_values) <= 1)])

# Plot
plt.figure(figsize=(10, 6))

plt.plot(n[np.isfinite(f2_values) & (np.abs(f2_values) <= 1)], arcsin_values2, label="arcsin2(f2(n))") #incoming?
plt.plot(n[np.isfinite(f2_values) & (np.abs(f2_values) <= 1)], np.pi-arcsin_values2, label="arcsin2(f2(n))") #incoming?

plt.plot(n[np.isfinite(f1_values) & (np.abs(f1_values) <= 1)], arcsin_values1, label="arcsin1(f1(n))") #outgoing?
plt.plot(n[np.isfinite(f1_values) & (np.abs(f1_values) <= 1)], np.pi-arcsin_values1, label="arcsin1(f1(n))") #outgoing?

plt.title("Plot of arcsin(f(n)) for n in [2, 100]")
plt.xlabel("n")
plt.ylabel("arcsin(f(n))")
plt.grid()
plt.legend()
plt.show()
