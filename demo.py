import random
import matplotlib.pyplot as plt

# 模拟16强的球队及其相对胜率（越高代表越强）
teams = {
    "Team A": 0.75,
    "Team B": 0.65,
    "Team C": 0.70,
    "Team D": 0.85,
    "Team E": 0.80,
    "Team F": 0.60,
    "Team G": 0.78,
    "Team H": 0.55,
    "Team I": 0.68,
    "Team J": 0.72,
    "Team K": 0.77,
    "Team L": 0.66,
    "Team M": 0.69,
    "Team N": 0.76,
    "Team O": 0.74,
    "Team P": 0.60
}


# 进行比赛模拟的函数
def simulate_match(team1, team2):
    prob1 = teams[team1]
    prob2 = teams[team2]

    # 计算两队胜出的概率
    total_prob = prob1 + prob2
    prob1 /= total_prob
    prob2 /= total_prob

    # 使用随机数决定胜者
    return team1 if random.random() < prob1 else team2


# 进行每轮的比赛并返回胜者
def simulate_round(teams_in_round):
    winners = []
    for i in range(0, len(teams_in_round), 2):
        winner = simulate_match(teams_in_round[i], teams_in_round[i + 1])
        winners.append(winner)
        print(f"{teams_in_round[i]} vs {teams_in_round[i + 1]} -> Winner: {winner}")
    return winners


# 预测从16强到决赛的过程
def predict_world_cup(teams):
    # 16强的第一轮
    print("\nQuarterfinals:")
    quarterfinals_winners = simulate_round(list(teams.keys()))

    # 四强的第二轮
    print("\nSemifinals:")
    semifinals_winners = simulate_round(quarterfinals_winners)

    # 决赛
    print("\nFinal:")
    winner = simulate_match(semifinals_winners[0], semifinals_winners[1])
    print(f"Winner of the World Cup: {winner}")

    # 返回最终的胜者
    return winner


# 显示预测过程的比赛进程图
def plot_predictions(teams):
    team_names = list(teams.keys())
    team_probs = list(teams.values())

    plt.figure(figsize=(10, 6))
    plt.barh(team_names, team_probs, color='skyblue')
    plt.xlabel("Winning Probability")
    plt.title("Team Strengths (Probability of Winning)")
    plt.show()


# 运行预测
plot_predictions(teams)
winner = predict_world_cup(teams)
