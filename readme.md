Image crawling / Scrapping through bing images  API and Google images crawling / Scrapping  using webdriver

1. Scrapping through Bing API :
    
    requirements:
    * Bing API key from microsoft portal
    * Open AI key

    procedure (How it works):
    i. takes the category to search from the category definition excel file
    ii. searches the images using the bing API URL
    iii. fetch images url , downlaods temporarily, resize image to 720p,checks for resolution and sharpness,verifies the image whether it matches the description and finally saves in the desired path

2. Scrapping through Google images using webdriver :

    Note:
    This code works for Google Chrome Version : 131.0.6778.86 (Official Build) (64-bit)
    Check ur version by : go to google > click 3 dots in top right corner > help > About google chrome
    
    requirements:
    * Check ur google chrome version and download the necessary driver in the link --> (https://developer.chrome.com/docs/chromedriver/downloads)
    * Open AI key

    What it is ? :
    This code use selenium chrome driver to automate the process of manual work which we used to download images from Google images. You can watch the process of doing this in the given youtube video --> (https://youtu.be/Yt6Gay8nuy0?si=P6Ks7eZuJrgIBKuh)
    
    procedure (How it works):
    i. takes the category to search from the category definition excel file
    ii. searches the images using the Google images URL
    iii. fetch images url , downlaods temporarily, resize image to 720p,checks for resolution and sharpness,verifies the image whether it matches the description and finally saves in the desired path