import os
import re
import traceback
import numpy as np
# import gpt_2_simple as gpt2
import lyricsgenius as lg


ACCESS_TOKEN = "V-JvkvIOA4D7PATFko0i8FfWP89pSper66mFCwFJ-BLSSE5B6vVD4RDt3MeWqFgP"
genius = lg.Genius(ACCESS_TOKEN,
                   verbose=True,
                   skip_non_songs=True,
                   excluded_terms=["(Remix)", "(Live)"],
                   remove_section_headers=True)

# File to which lyrics are written to
FILE_NAME = "lyrics.txt"

ARTISTS = ['Drake']


def download_model(model_name):
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)

def add_breaks(lyrics):
    lyrics = lyrics.replace("\n\n\n", "\n\n")
    verses_with_breaks = list(map(lambda x : x.replace("\n", " <|LINE_BREAK|>\n"), 
                        lyrics.split("\n\n")))
    return "\n<|VERSE_BREAK|>\n".join(verses_with_breaks)


def get_lyrics(artists, max_songs, lyrics_file):
    """
    Retrieves lyrics using Genius API wrapper (lyricsgenius),
    saves the results in a text file

    :param artists: list of artists to find lyrics for
    :param max_songs: number of songs to search for per artist
    :param lyrics_file: all lyrics will be written to this file
    """
    with open(lyrics_file, 'w') as f:
        for artist in artists:
            try:
                songs_obj = (genius.search_artist(
                    artist, max_songs=max_songs, sort="title")).songs

                # compress the lyrics into a single string
                songs = [song.lyrics for song in songs_obj]

                songs_with_breaks = map(add_breaks, songs)

                start_delim = "\n\n<|startoftext|>\n\n"
                end_delim = "\n\n<|endoftext|>\n\n"
                song_delim = end_delim + start_delim

                f.write(start_delim)
                f.write(song_delim.join(songs_with_breaks))
                f.write(end_delim)

                print(f"Grabbed {len(songs)} songs from {artist}")
            except:
                print(f"Exception at {artist}")
                traceback.print_exc()

def get_data(train_file, test_file):
    with open(train_file, 'r') as f:
        train = f.read().strip().split()

    with open(test_file, 'r') as f:
        test = f.read().strip().split()

    vocab = dict()
    counter = 0
    train_ind = []
    for word in train:
        if word not in vocab:
            vocab[word] = counter
            counter += 1
        train_ind.append(vocab[word])
    
    test_ind = []
    for word in test:
        test_ind.append(vocab[word])
    
    return train_ind, test_ind, vocab


def main():
    get_lyrics(ARTISTS, 100, "lyrics.txt")

    # Download the model locally
    # model_name = "124M"
    # download_model(model_name)

    # sess = gpt2.start_tf_sess()
    # gpt2.finetune(sess,
    #   FILE_NAME,
    #   model_name = model_name,
    #   steps = 1000)   # steps is max number of training steps

    # gpt2.generate(sess,
    #   prefix="<|startoftext|>",
    #   truncate="<|endoftext|>"
    #   )


if __name__ == '__main__':
    main()
