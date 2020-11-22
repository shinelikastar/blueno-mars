import os
import re
import traceback
import numpy as np
import gpt_2_simple as gpt2
import lyricsgenius as lg


ACCESS_TOKEN = "V-JvkvIOA4D7PATFko0i8FfWP89pSper66mFCwFJ-BLSSE5B6vVD4RDt3MeWqFgP"
genius = lg.Genius(ACCESS_TOKEN,
                   verbose=False,
                   skip_non_songs=True,
                   excluded_terms=["(Remix)", "(Live)"],
                   remove_section_headers=True)

# File to which lyrics are written to
FILE_NAME = "lyrics.txt"
SAVED_LYRICS = open(FILE_NAME, "w")

# List of artists to scrape lyrics for
ARTISTS = ['Drake']


def download_model(model_name):
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)


def get_lyrics(artists, max_songs, lyrics_file):
    """
    Retrieves lyrics using Genius API wrapper (lyricsgenius),
    saves the results in a text file 

    :param artists: list of artists to find lyrics for
    :param max_songs: number of songs to search for per artist 
    :param lyrics_file: all lyrics will be written to this file
    """
    for artist in artists:
        try:
            songs_obj = (genius.search_artist(
                artist, max_songs=max_songs, sort="title")).songs

            # compress the lyrics into a single string
            lyrics = [song.lyrics for song in songs_obj]

            start_delim = "\n \n   <|startoftext|>   \n \n"
            end_delim = "\n \n   <|endoftext|>   \n \n"

            song_delim = end_delim + start_delim

            lyrics_file.write(start_delim)
            lyrics_file.write(song_delim.join(lyrics))
            lyrics_file.write(end_delim)

            print(f"Grabbed {len(lyrics)} songs from {artist}")
        except:
            print(f"Exception at {artist}")
            traceback.print_exc()


def main():
    get_lyrics(ARTISTS, 10, SAVED_LYRICS)

    # Download the model locally
    model_name = "124M"
    download_model(model_name)

    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                  FILE_NAME,
                  model_name=model_name,
                  steps=1000)   # steps is max number of training steps

    gpt2.generate(sess, 
                  prefix="<|startoftext|>",
                  truncate="<|endoftext|>"
                  )


if __name__ == '__main__':
    main()
