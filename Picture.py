import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

df = pd.read_csv("samples_vs_values.csv")
df = df.drop(index=0)


samples = df["Unnamed: 0"]
with_opt = df["Option"]
no_opt = df["No option"]

sigma_opt = df["Option dev"]
sigma_no_opt = df["No option dev"]

plt.plot(samples, with_opt, label="with option")
plt.plot(samples, no_opt, label="no option")
plt.xlabel("Samples")
plt.ylabel("Estimate")
plt.legend()
plt.title("Lower bound estimate vs samples")
plt.show()

plt.plot(samples, sigma_opt, label="with option")
plt.plot(samples, sigma_no_opt, label="no option")
plt.xlabel("Samples")
plt.ylabel("Estimate")
plt.legend()
plt.title("Standard deviation vs samples")
plt.show()

a=1