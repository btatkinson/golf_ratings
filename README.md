# Golf Ratings

### Purpose

In this repo I compare an improved Elo system to Glicko on golf data. The golf data is PGA Tour and European tour data since 2000. I used my [golf scraper](https://github.com/btatkinson/golf_scraper) to collect it. A writeup of the results is available on [Medium](https://medium.com/@BlakeAtkinson/rating-sports-teams-maximizing-a-generic-system-772144574a07?postPublishedType=initial).

### Data

The path I use for loading tournament leaderboards is: '../golf_scraper/leaderboards/'+season+'/'+tour+'/'+tournament_name+'.csv'. Each leaderboard contains the round by round scores of each golfer. It also contains a date string in the datetime format of '%b %d %Y'. Since the tours don't provide unique ids, I'm forced to use the golfer name as an ID. The issue there is that two golfers with the same name will corrupt each other's ratings. I also have to translate names from European Tour to PGA Tour so that they are consistent. I didn't spend a ton of time translating names, and so only good players have their names translated. Otherwise, players will carry two ratings: one PGA and one Euro.

### Packages

I don't use any exotic packages, I think all are standard python data science tools. The packages I use to plot might be unique, especially [seaborn](https://seaborn.pydata.org/).

** Update June 28th 2019 **
I use the Map class to find distances between two tournaments. That class has googlemaps (which requires an API key). For reasons discussed in [this article](https://medium.com/@BlakeAtkinson/do-long-flights-affect-professional-golfers-1995699d5611?postPublishedType=repub), I moved my location data tracking to ./old_files/main.py and ./old_files/analyze.py

### Running
The biggest challenge to replicating this repo yourself will be supplying the data. I tried to comment the code as best I could, but as this was a personal hobby project, I did take shortcuts in a couple of spots and left commented out code in some areas. ``` main.py ``` contains almost all of the logic needed. If you see a function you don't recognize, it probably exists in ```helpers.py```. I use classes described in the classes folder, including Elo, Glicko, and Player. ``` python3 main.py ``` is what I use to execute.

I separated out the file to create graphs. That's currently in ```./old_files/analyze.py```. It's not fun to parse through, because it's full of temporary code meant for discard. I kept it as a personal reference.

I quickly wrote ```predict.py``` to predict matchups in upcoming PGA events. It's far from perfect but it's a good start to using the ratings.

Parameters can be adjusted in ```settings.py```.
