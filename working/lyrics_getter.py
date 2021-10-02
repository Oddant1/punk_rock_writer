import json
from azlyrics import artists, songs, lyrics

discography = json.loads(songs('badreligion'))
albums = discography['albums']
for album in albums:
    for song in albums[album]:
        text = lyrics('badreligion', song)
        for line in text:
            print(line)
