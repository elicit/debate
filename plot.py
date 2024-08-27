import matplotlib.pyplot as plt
import numpy as np

# DebateAgent-gpt-3.5-turbo-0125: -82.75610096777383 (-97.99992222574704, -69.75180282935165)
# BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4: -71.35432775571576 (-84.39398853674521, -58.30688130079728)
# BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4: 9.90872110229842 (-2.5111131966985143, 23.997587006863068)
# DebateAgent-gpt-4o-mini-2024-07-18: 13.379658024392112 (-1.0005080065253045, 26.809333170637256)
# BoNDebateAgent-gpt-4o-2024-05-13-best_of-4: 63.92375111408652 (52.31658753566865, 77.36733839236162)
# DebateAgent-gpt-4o-2024-05-13: 66.89059954552866 (54.22025813147223, 79.9829901725828)

# Blind judge accuracy: 0.692 CI: (0.672, 0.712)
# Agent: DebateAgent-gpt-3.5-turbo-0125 Accuracy: 0.6555 CI: (0.635, 0.6765)
# Agent: BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4 Accuracy: 0.6295 CI: (0.608, 0.65)
# Agent: DebateAgent-gpt-4o-mini-2024-07-18 Accuracy: 0.709 CI: (0.689, 0.729)
# Agent: BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4 Accuracy: 0.6905 CI: (0.67, 0.7105)
# Agent: DebateAgent-gpt-4o-2024-05-13 Accuracy: 0.733 CI: (0.7135, 0.7525)
# Agent: BoNDebateAgent-gpt-4o-2024-05-13-best_of-4 Accuracy: 0.752 CI: (0.7335, 0.7705)
# Data
agents = [
    "GPT-3.5",
    "GPT-3.5 best of 4",
    "GPT-4o mini",
    "GPT-4o mini best of 4",
    "GPT-4o",
    "GPT-4o best of 4",
]

accuracies = [0.6555, 0.6295, 0.709, 0.6905, 0.733, 0.752]
accuracy_lower = [0.635, 0.608, 0.689, 0.67, 0.7135, 0.7335]
accuracy_upper = [0.6765, 0.65, 0.729, 0.7105, 0.7525, 0.7705]

elo_ratings = [-82.75610096777383, -71.35432775571576, 13.379658024392112, 9.90872110229842, 66.89059954552866, 63.92375111408652]
elo_lower = [-97.99992222574704, -84.39398853674521, -1.0005080065253045, -2.5111131966985143, 54.22025813147223, 52.31658753566865]
elo_upper = [-69.75180282935165, -58.30688130079728, 26.809333170637256, 23.997587006863068, 79.9829901725828, 77.36733839236162]

# Create the plot
plt.figure(figsize=(12, 8))

# Plot points with error bars
plt.errorbar(elo_ratings, accuracies, 
             xerr=[np.array(elo_ratings) - np.array(elo_lower), np.array(elo_upper) - np.array(elo_ratings)],
             yerr=[np.array(accuracies) - np.array(accuracy_lower), np.array(accuracy_upper) - np.array(accuracies)],
             fmt='o', color='blue', ecolor='lightblue', capsize=5, label='Agents')

# Add labels for each point
for i, agent in enumerate(agents):
    plt.annotate(
        agent,
        (elo_ratings[i], accuracies[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# Add blind judge line with confidence interval
plt.axhline(y=0.692, color="gray", linestyle="--", label="Blind Judge (69.2%)")

# Calculate and add line of best fit
z = np.polyfit(elo_ratings, accuracies, 1)
p = np.poly1d(z)
plt.plot(
    elo_ratings,
    p(elo_ratings),
    "r--",
    label="Line of Best Fit",
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

# Print the equation of the line of best fit
print(f"Line of Best Fit Equation: y = {z[0]:.6f}x + {z[1]:.6f}")