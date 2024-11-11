import matplotlib.pyplot as plt
import numpy as np

# Data for base GPT-3.5 model
base_agents = [
    "GPT-3.5",
    "GPT-3.5 best of 4",
    "GPT-4o mini",
    "GPT-4o mini best of 4",
    "GPT-4o",
    "GPT-4o best of 4",
]

base_accuracies = [0.5655, 0.5690999291282778, 0.584, 0.585, 0.5815, 0.607]
base_accuracy_lower = [0.5435, 0.5428773919206237, 0.5625, 0.563, 0.56, 0.586]
base_accuracy_upper = [0.5875, 0.595322466335932, 0.606, 0.6065, 0.603, 0.6285]

base_elo_ratings = [
    -138.01726328138918,
    -126.23790522730957,
    27.607497829729297,
    59.470244793716915,
    69.28563864451445,
    107.87898475905843,
]
base_elo_lower = [
    -154.19661512840582,
    -140.4473038384231,
    13.771650924952626,
    45.53694734678249,
    55.34562604774892,
    93.0833721778311,
]
base_elo_upper = [
    -122.41891714502053,
    -112.19360774749182,
    40.815277305859716,
    73.12523558085847,
    82.8199152313746,
    123.23623415798153,
]

# Data for fine-tuned model
ft_agents = [
    "GPT-3.5",
    "GPT-3.5 best of 4",
    "GPT-4o mini",
    "GPT-4o mini best of 4",
    "GPT-4o",
    "GPT-4o best of 4",
]

ft_accuracies = [0.6555, 0.6295, 0.709, 0.6905, 0.733, 0.752]
ft_accuracy_lower = [0.635, 0.608, 0.689, 0.67, 0.7135, 0.7335]
ft_accuracy_upper = [0.6765, 0.65, 0.729, 0.7105, 0.7525, 0.7705]

ft_elo_ratings = [
    -82.75610096777383,
    -71.35432775571576,
    13.379658024392112,
    9.90872110229842,
    66.89059954552866,
    63.92375111408652,
]
ft_elo_lower = [
    -97.99992222574704,
    -84.39398853674521,
    -1.0005080065253045,
    -2.5111131966985143,
    54.22025813147223,
    52.31658753566865,
]
ft_elo_upper = [
    -69.75180282935165,
    -58.30688130079728,
    26.809333170637256,
    23.997587006863068,
    79.9829901725828,
    77.36733839236162,
]

# Create the plot
plt.figure(figsize=(14, 10))

# Plot points with error bars for base model
plt.errorbar(
    base_elo_ratings,
    base_accuracies,
    xerr=[
        np.array(base_elo_ratings) - np.array(base_elo_lower),
        np.array(base_elo_upper) - np.array(base_elo_ratings),
    ],
    yerr=[
        np.array(base_accuracies) - np.array(base_accuracy_lower),
        np.array(base_accuracy_upper) - np.array(base_accuracies),
    ],
    fmt="o",
    color="purple",
    ecolor="plum",
    capsize=5,
    label="Base GPT-3.5 Judge",
)

# Plot points with error bars for fine-tuned model
plt.errorbar(
    ft_elo_ratings,
    ft_accuracies,
    xerr=[
        np.array(ft_elo_ratings) - np.array(ft_elo_lower),
        np.array(ft_elo_upper) - np.array(ft_elo_ratings),
    ],
    yerr=[
        np.array(ft_accuracies) - np.array(ft_accuracy_lower),
        np.array(ft_accuracy_upper) - np.array(ft_accuracies),
    ],
    fmt="o",
    color="blue",
    ecolor="lightblue",
    capsize=5,
    label="Fine-tuned GPT-3.5 Judge",
)

# Add labels for each point
for i, agent in enumerate(base_agents):
    plt.annotate(
        agent,
        (base_elo_ratings[i], base_accuracies[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        color="purple",
    )
    plt.annotate(
        agent,
        (ft_elo_ratings[i], ft_accuracies[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        color="blue",
    )

# Add blind judge line with confidence interval
plt.axhline(y=0.692, color="gray", linestyle="--", label="Blind Judge (69.2%)")

# Calculate and add lines of best fit
base_z = np.polyfit(base_elo_ratings, base_accuracies, 1)
base_p = np.poly1d(base_z)
plt.plot(
    base_elo_ratings,
    base_p(base_elo_ratings),
    "r--",
    label="Base GPT-3.5 Judge Best Fit",
    color="purple",
    alpha=0.5,
)

ft_z = np.polyfit(ft_elo_ratings, ft_accuracies, 1)
ft_p = np.poly1d(ft_z)
plt.plot(
    ft_elo_ratings,
    ft_p(ft_elo_ratings),
    "b--",
    label="Fine-tuned GPT-3.5 Judge Best Fit",
    color="blue",
    alpha=0.5,
)

# Set labels and title
plt.xlabel("Elo Rating")
plt.ylabel("Accuracy")
plt.title("GPT-3.5 Judge MMLU Pro Accuracy vs Elo Rating")

# Add legend
plt.legend()

# Show grid
plt.grid(True, linestyle=":", alpha=0.7)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show the plot
plt.show()

# Print the equations of the lines of best fit
print(f"Base Model Line of Best Fit Equation: y = {base_z[0]:.6f}x + {base_z[1]:.6f}")
print(f"Fine-tuned Model Line of Best Fit Equation: y = {ft_z[0]:.6f}x + {ft_z[1]:.6f}")
