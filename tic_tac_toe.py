from base import Game, OptimalValueEstimator, RandomPlayer, ValuePlayer, gather_stats, simulate_game


class TicTacToeGame(Game):
    LINES = [0b000000111, 0b000111000, 0b111000000, 0b001001001, 0b010010010, 0b100100100, 0b100010001, 0b001010100]

    @classmethod
    def starting_position(cls):
        return (0, 0)

    @classmethod
    def fmt(cls, board):
        my, other = board
        return " ".join("".join("X" if my & (1 << i) else "O" if other & (1 << i) else "." for i in range(j, j+3)) for j in range(0, 9, 3))

    @classmethod
    def legal_moves(cls, board):
        my, other = board
        whole = my | other
        if whole == 0b111111111:
            return [], 0
        for line in cls.LINES:
            if my & line == line:
                return [], 1
            if other & line == line:
                return [], -1
        return [i for i in range(9) if whole & (1 << i) == 0], None

    @classmethod
    def play_move(cls, board, idx):
        my, other = board
        whole = my | other
        if whole & (1 << idx):
            raise ValueError(f"Cell {idx} is not empty")
        return (other, my | (1 << idx))


def main():
    n_games = 1000
    random_player = RandomPlayer(TicTacToeGame)
    optimal_player = ValuePlayer(TicTacToeGame, OptimalValueEstimator(TicTacToeGame))
    win, draw, loss = gather_stats(TicTacToeGame, optimal_player, random_player, n_games)
    print(f"Out of {n_games} games:")
    print(f"Optimal wins: {win} ({win/n_games*100:.1f}%)")
    print(f"Random wins: {loss} ({loss/n_games*100:.1f}%)")
    print(f"Draws: {draw} ({draw/n_games*100:.1f}%)") 


if __name__ == '__main__':
    main()
