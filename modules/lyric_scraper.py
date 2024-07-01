import requests
from bs4 import BeautifulSoup
from config import genius_access_token

access_token = genius_access_token


def search_song(song_name):
    # Replace 'YOUR_ACCESS_TOKEN' with your Genius API access token
    headers = {"Authorization": "Bearer " + access_token}
    base_url = "https://api.genius.com"
    search_url = base_url + "/search?per_page=10&page=1&q=" + song_name

    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Check if any hits were found
        if data["meta"]["status"] == 200 and data["response"]["hits"]:
            # Extract the first hit
            hit = data["response"]["hits"][0]["result"]
            # Print the page URL
            return hit["url"]
        else:
            return None
    except requests.exceptions.RequestException as e:
        print("Error:", e)


def scrape_song_lyrics(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    # artist_name = (
    #     soup.select('div[class*="HeaderArtistAndTracklistdesktop__Container-sc-"]')[0]
    #     .select('a[class*="HeaderArtistAndTracklistdesktop__Artist-sc-"]')[0]
    #     .get_text()
    # )
    # song_name = soup.select('h1[class*="SongHeaderdesktop__Title-sc-"]')[0].get_text()

    # print(f"Scraping {song_name} by {artist_name}...")

    lyrics = soup.select('div[class*="Lyrics__Container-sc"]')
    lyrics = "\n".join([lyric.get_text(separator="\n") for lyric in lyrics])

    # print(lyrics)

    # return {"song_name": song_name, "artist_name": artist_name, "lyrics": lyrics}
    return lyrics


def get_lyrics(song_name):
    url = search_song(song_name)
    if url:
        lyrics = scrape_song_lyrics(url)
        return lyrics
    else:
        # return {"song_name": "", "artist_name": "", "lyrics": ""}
        return None

    # TODO: Verify if the lyrics are fetched from the correct song
