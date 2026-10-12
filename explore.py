# import the necessary libraries
import zstandard as zstd
import io
import chess.pgn

file_path = "data/lichess_2024_01_partial.pgn.zst"

with open(file_path, "rb") as f:
    # ZstdDecompressor is used to decompress the zst file
    dctx = zstd.ZstdDecompressor()

    # stream_reader is used to read the decompressed binary file as a stream
    stream_reader = dctx.stream_reader(f)

    # TextIOWrapper converts the binary data to a text data
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

    for _ in range(10):
        # chess.pgn.read_game reads a line one by one from the stream until it has read a game completely and then parses it
        game = chess.pgn.read_game(text_stream)
        
        print(game)
