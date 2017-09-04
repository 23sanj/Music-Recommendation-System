Offline Music Recommendation System:
This project is my initial attempt to recommend similar artists to users.

Uses item based collaborative filtering technique. Since users' listening trend is dynamic, user based filtering is avoided.
The dataset is collected from Millions songs dataset, we have three files:
1. user_file 
2. song_file
3. artist file


Implicit ratings in the form of player listen count is taken and the sparse UxI matrix is built.

To make recommendations, the model uses k-nearest neighbors technique to output recommendations.
Cosine distance is used as the metric.A query_artist is input to the model, and the model uses:
fuzzy matching(approximate string matching--ratio: 70%) to search for the input artist in the database, and outputs 10 similar artists as recommendations. 

In addition, the model also outputs the cover art of the artists' album. It's a two step process:
Used the musicbrainz api to pull the most recent releases of each similar artist, using the mbid of the artist from the artist file.
Pulled a random cover_art jpg image for a release from the response generated from the coverartarchive.org 

By limiting to 50% popular artists, the noise in the input sparse matrix to the model is significantly reduced.

Libraries explored:

Data Preprocessing using Pandas and numpy
scikit to build the model
fuzzywuzzy to input query artist
Ipython image to display images
requests to pull xml data from apis
Elementree to parse xml files


