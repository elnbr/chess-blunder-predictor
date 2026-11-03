# import the necessary libraries
import zstandard as zstd
import io
import chess.pgn
import re
import numpy as np
import csv


def main():
    filtered_games = filter_games(path="data/lichess_2024_01_partial.pgn.zst", n_games=5_000)

    train, groups = get_moves(games=filtered_games)

    with open("data/train.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(train)

    with open("data/game_id.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(groups)


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


def get_moves(games: list[chess.pgn.Game]) -> tuple[list[np.ndarray], list[str]]:
    all_moves = []
    train = []
    groups = []

    for game_id, game in enumerate(games):
        # get the time control
        tc = game.time_control()

        if not tc.parts:
            continue

        base = tc.parts[0].time
        increment = tc.parts[0].increment if tc.parts[0].increment is not None else 0

        # get the player elos
        white_elo = int(game.headers.get("WhiteElo"))
        black_elo = int(game.headers.get("BlackElo"))
        
        for i, node in enumerate(game.mainline()):
            # start reading the moves after move 5 because the first few moves are mainly theory
            if i < 10:
                continue
            
            parent_node = node.parent
            
            # the side that is moving right now (returns True for white and False for black)
            moving_side = parent_node.turn()

            # the time that the moving side had in seconds before they played their move
            moving_side_clock = parent_node.parent.clock()

            if moving_side_clock is None:
                continue

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
                "moving_side": int(moving_side),
                "is_blunder": is_blunder,
                "moving_side_clock": moving_side_clock,
                "base_time": base,
                "increment": increment,
                "player_elo": white_elo if int(moving_side) else black_elo,
                "opponent_elo": black_elo if int(moving_side) else white_elo,
                "game_id": game_id
            }

            all_moves.append(move_delta)

    for dict in all_moves:

        # we need the board_array in order to OHE every single piece
        board_array = fen_to_board(dict["position_before"])

        # OHE every square for each unique piece. this leads to 12 * 64 = 768 features (# of unique pieces * total squares)
        piece_OHE = board_to_OHE(board_array)
        # normalize the remaining features and add them
        row = np.concatenate([
            piece_OHE, 
            np.array([dict["moving_side"], 
                      dict["moving_side_clock"] / (dict["base_time"] + 40 * dict["increment"]), 
                      dict["base_time"] / 300,
                      dict["increment"] / 5, 
                      (dict["player_elo"] - 1500) / 500,
                      (dict["opponent_elo"] - 1500) / 500,
                      dict["is_blunder"]])
        ]).astype(np.float32)

        train.append(row)

        groups.append(dict["game_id"])

    return train, groups


def fen_to_board(fen: str) -> np.ndarray:
    piece_to_number = {
        "P": 1,
        "N": 2,
        "B": 3,
        "R": 4,
        "Q": 5,
        "K": 6,
        "p": -1,
        "n": -2,
        "b": -3,
        "r": -4,
        "q": -5,
        "k": -6
    }

    board_array = np.array([])

    board = chess.Board(fen)

    # store every piece position in a 1d array and map pieces to numbers. positive numbers for white pieces, negative for black pieces, and 0 for empty squares
    for square in chess.SQUARES:
        board_array = np.append(
            board_array, 
            piece_to_number[str(board.piece_at(square))] if board.piece_at(square) is not None else 0
            )

    return board_array


def board_to_OHE(board_array: np.ndarray) -> np.ndarray:
    piece_numbers = np.array([[1],
                              [2],
                              [3],
                              [4],
                              [5],
                              [6],
                              [-1],
                              [-2],
                              [-3],
                              [-4],
                              [-5],
                              [-6]])

    # kron will generate a 2d array where each element is a board array
    # each board array will have been multiplied by some scalar value (e.g. first board array is multiplied by 1 and last one is multipled by -6)
    kron_array = np.kron(piece_numbers, board_array)

    # this is how we are going to OHE each square for each piece
    # this will result in 12 * 64 features total (# of unique pieces * total squares)
    for i in range(len(kron_array)):
        kron_array[i] = (kron_array[i] == (piece_numbers[i][0] * piece_numbers[i][0])).astype(np.uint8)

    # turn the 2d array into a 1d array and return it
    return kron_array.ravel().astype(np.uint8)


def calculate_win_percent(centipawns: int) -> float:
    # source for formula: https://lichess.org/page/accuracy
    return (50 + 50 * (2 / (1 + np.exp(-0.00368208 * centipawns)) - 1))


if __name__ == "__main__":
    main()
