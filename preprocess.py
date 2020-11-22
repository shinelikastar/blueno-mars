import lyricsgenius as lg
import traceback
import numpy as np
import re

ACCESS_TOKEN = "V-JvkvIOA4D7PATFko0i8FfWP89pSper66mFCwFJ-BLSSE5B6vVD4RDt3MeWqFgP"
genius = lg.Genius(ACCESS_TOKEN,
                   verbose=False,
                   skip_non_songs=True,
                   excluded_terms=["(Remix)", "(Live)"],
                   remove_section_headers=True)

# File to write lyrics to
SAVED_LYRICS = open("lyrics.txt", "w")

# List of artists to scrape lyrics for
ARTISTS = ['Drake']


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

            lyrics = [song.lyrics for song in songs_obj]

            # compress the lyrics into a single string
            end_of_song_delim = "\n \n   <|endoftext|>   \n \n"
            lyrics_file.write(end_of_song_delim.join(lyrics))
            lyrics_file.write(end_of_song_delim)

            print(f"Grabbed {len(lyrics)} songs from {artist}")
        except:
            print(f"Exception at {artist}")
            traceback.print_exc()


def main():
    get_lyrics(ARTISTS, 3, SAVED_LYRICS)


if __name__ == '__main__':
    main()
