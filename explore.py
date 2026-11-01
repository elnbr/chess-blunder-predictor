# import the necessary libraries
import zstandard as zstd
import io
import chess.pgn
import re
import numpy as np


def main():
    filtered_games = filter_games(path="data/lichess_2024_01_partial.pgn.zst", n_games=1)

    get_moves(games=filtered_games)


def filter_games(path: str, n_games: int) -> list[chess.pgn.Game]:

    with open(path, "rb") as f:
        # ZstdDecompressor is used to decompress the zst file
        dctx = zstd.ZstdDecompressor()

        # stream_reader is used to read the decompressed binary file as a stream
        stream_reader = dctx.stream_reader(f)

        # TextIOWrapper converts the binary data to a text data
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

        filtered_games = []
        curr_game_lines = []
        
        for line in text_stream:
            if len(filtered_games) >= n_games:
                break

            if line.startswith("[Event "):
                game = "".join(curr_game_lines)

                if game and passes_conditions(game=game):
                    filtered_games.append(chess.pgn.read_game(io.StringIO(game)))
                
                curr_game_lines = []

            curr_game_lines.append(line)
            
        return filtered_games


def passes_conditions(game: str) -> bool:
    white_elo = re.search(r'WhiteElo "(\d+)"', game)
    black_elo = re.search(r'BlackElo "(\d+)"', game)

    if white_elo is None or black_elo is None:
        return False

    white_elo = white_elo.group(1)
    black_elo = black_elo.group(1)
    
    if (
        ("[%eval" in game) and
        ("Rated Blitz " in game) and
        (1000 <= int(white_elo) <= 2000) and
        (1000 <= int(black_elo) <= 2000)
    ):
        return True


def get_moves(games: list[chess.pgn.Game]):
    move_delta = {}
    all_moves = []

    for game in games:
        for i, node in enumerate(game.mainline()):
            # start reading the moves after move 5 because the first few moves are mainly theory
            if i < 10:
                continue
            
            parent_node = node.parent

            # the side that is moving right now
            moving_side = parent_node.turn()

            # the time that the moving side had before they played their move
            moving_side_clock = parent_node.parent.clock()

            # skip evals that are None
            if parent_node.eval() is None or node.eval() is None:
                continue
            
            centipawns_before = parent_node.eval().pov(moving_side).score(mate_score=10000)
            # the win percent before the player made their move
            win_percent_before = calculate_win_percent(centipawns_before)

            centipawns_after = node.eval().pov(moving_side).score(mate_score=10000)
            # the win percent after the player made their move
            win_percent_after = calculate_win_percent(centipawns_after)

            drop = win_percent_before - win_percent_after
            is_blunder = 1 if drop >= 15 else 0

            move_delta = {
                "position_before": parent_node.board().fen(),
                "win_percent_before": win_percent_before,
                "win_percent_after": win_percent_after,
                "drop": drop,
                "is_blunder": is_blunder,
                "time_remaining_before_move": moving_side_clock
            }

            all_moves.append(move_delta)

    for dict in all_moves:
        for key, value in dict.items():
            print(f"{key}: {value}")
        print()


def calculate_win_percent(centipawns: int) -> float:
    # source for formula: https://lichess.org/page/accuracy
    return (50 + 50 * (2 / (1 + np.exp(-0.00368208 * centipawns)) - 1))


if __name__ == "__main__":
    main()
