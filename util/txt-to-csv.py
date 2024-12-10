# This script produces a csv with just these categories as columns.
# Given a text file with the format:
# "- Artist - Song Name, Genre, Label, ..."

import csv
import re

def create_data(txt, arr):
    # Read text file
    with open(txt, 'r') as file:
        for line in file:
            row = line.strip()[2:]
            
            # Artist
            artist = re.search('^[^-]*', row).group(0)
            artist = artist[:len(row)-1]
            row = re.search('-(.*)', row).group(0)
            row = row[2:]
            # Song Name
            song_name = row
            song_name = re.search('^[^,]*', song_name).group(0)
            row = re.search(',(.*)', row).group(0)
            row = row[1:]
            # Genre
            genre = row
            genre = re.search('^[^,]*', genre).group(0)
            row = re.search(',(.*)', row).group(0)
            row = row[1:]
            # Label
            label = row
            label = re.search('^[^,]*', label).group(0)
            row = re.search(',(.*)', row).group(0)
            row = row[1:]
            # Other
            other = row

            """
            Printing Row Info:

            print("Artist: " + artist)
            print("Song Name: " + song_name)
            print("Genre: " + genre)
            print("Label: " + label)
            print("Other: " + other)
            """

            # Adding info to data table
            arr.append([artist, song_name, genre, label, other])


def convert_data(name, arr):
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(arr)


if __name__ == "__main__":
    TEXT_FILE_NAME = "text_files/message.txt"
    data = [
        ['Artist', 'Song Name', 'Genre', 'Label', 'Other']
    ]

    create_data(TEXT_FILE_NAME, data)

    convert_data('song_data.csv', data)
