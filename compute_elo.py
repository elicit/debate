import math

from scipy.optimize import minimize


# Calculate expected score
def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + math.pow(10, (rating_b - rating_a) / 500))


def compute_negative_log_likelihood(
    win_rates: dict[tuple[str, str], float],
    elo_ratings: dict[str, float],
):
    """Compute the negative log-likelihood loss between the predicted win rates and the actual win rates."""
    nll_loss: float = 0
    for (player_a, player_b), actual_win_rate in win_rates.items():
        predicted_win_rate: float = expected_score(
            elo_ratings[player_a], elo_ratings[player_b]
        )
        nll_loss += -actual_win_rate * math.log(predicted_win_rate)
    return nll_loss


def compute_elo_with_optimization(
    win_rates: dict[tuple[str, str], float],
    base_rating: float = 0,
):
    """Compute Elo ratings using numerical optimization."""
    players: list[str] = []
    for player_a, player_b in win_rates.keys():
        players.append(player_a)
        players.append(player_b)

    def objective_function(x):
        elo_ratings: dict[str, float] = {
            player: x[i] for i, player in enumerate(players)
        }
        return compute_negative_log_likelihood(win_rates, elo_ratings)

    # Initial Elo ratings
    initial_elo_ratings: list[float] = [base_rating] * len(players)

    # Perform optimization
    result = minimize(objective_function, initial_elo_ratings, method="BFGS")
    elo_ratings: dict[str, float] = {
        player: result.x[i] for i, player in enumerate(players)
    }
    return elo_ratings


# Example usage
win_rates: dict[tuple[str, str], float] = {
    ("DebateAgent-gpt-3.5-turbo-0125", "DebateAgent-gpt-3.5-turbo-0125"): 0.5,
    (
        "DebateAgent-gpt-3.5-turbo-0125",
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
    ): 0.455,
    ("DebateAgent-gpt-3.5-turbo-0125", "DebateAgent-gpt-4o-mini-2024-07-18"): 0.34,
    (
        "DebateAgent-gpt-3.5-turbo-0125",
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
    ): 0.325,
    ("DebateAgent-gpt-3.5-turbo-0125", "DebateAgent-gpt-4o-2024-05-13"): 0.31,
    (
        "DebateAgent-gpt-3.5-turbo-0125",
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
    ): 0.245,
    (
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
        "DebateAgent-gpt-3.5-turbo-0125",
    ): 0.545,
    (
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
    ): 0.5,
    (
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
        "DebateAgent-gpt-4o-mini-2024-07-18",
    ): 0.34,
    (
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
    ): 0.325,
    (
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
        "DebateAgent-gpt-4o-2024-05-13",
    ): 0.3,
    (
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
    ): 0.315,
    ("DebateAgent-gpt-4o-mini-2024-07-18", "DebateAgent-gpt-3.5-turbo-0125"): 0.66,
    (
        "DebateAgent-gpt-4o-mini-2024-07-18",
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
    ): 0.66,
    ("DebateAgent-gpt-4o-mini-2024-07-18", "DebateAgent-gpt-4o-mini-2024-07-18"): 0.5,
    (
        "DebateAgent-gpt-4o-mini-2024-07-18",
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
    ): 0.435,
    ("DebateAgent-gpt-4o-mini-2024-07-18", "DebateAgent-gpt-4o-2024-05-13"): 0.46,
    (
        "DebateAgent-gpt-4o-mini-2024-07-18",
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
    ): 0.38,
    (
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
        "DebateAgent-gpt-3.5-turbo-0125",
    ): 0.675,
    (
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
    ): 0.675,
    (
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
        "DebateAgent-gpt-4o-mini-2024-07-18",
    ): 0.565,
    (
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
    ): 0.5,
    (
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
        "DebateAgent-gpt-4o-2024-05-13",
    ): 0.49,
    (
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
    ): 0.42,
    ("DebateAgent-gpt-4o-2024-05-13", "DebateAgent-gpt-3.5-turbo-0125"): 0.69,
    (
        "DebateAgent-gpt-4o-2024-05-13",
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
    ): 0.7,
    ("DebateAgent-gpt-4o-2024-05-13", "DebateAgent-gpt-4o-mini-2024-07-18"): 0.54,
    (
        "DebateAgent-gpt-4o-2024-05-13",
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
    ): 0.51,
    ("DebateAgent-gpt-4o-2024-05-13", "DebateAgent-gpt-4o-2024-05-13"): 0.5,
    (
        "DebateAgent-gpt-4o-2024-05-13",
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
    ): 0.425,
    (
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
        "DebateAgent-gpt-3.5-turbo-0125",
    ): 0.755,
    (
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
        "BoNDebateAgent-gpt-3.5-turbo-0125-best_of-4",
    ): 0.685,
    (
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
        "DebateAgent-gpt-4o-mini-2024-07-18",
    ): 0.62,
    (
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
        "BoNDebateAgent-gpt-4o-mini-2024-07-18-best_of-4",
    ): 0.58,
    (
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
        "DebateAgent-gpt-4o-2024-05-13",
    ): 0.575,
    (
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
        "BoNDebateAgent-gpt-4o-2024-05-13-best_of-4",
    ): 0.5,
}

elo_ratings: dict[str, float] = compute_elo_with_optimization(win_rates)

print("Calculated Elo Ratings:")
for player, rating in sorted(elo_ratings.items(), key=lambda x: x[1]):
    print(f"{player}: {rating:.2f}")
