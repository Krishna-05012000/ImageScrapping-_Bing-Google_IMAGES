import os
import requests
from PIL import Image
# from pytesseract import image_to_string
import base64
import io
from openai import OpenAI
import cv2
import pytesseract
import pandas as pd
import multiprocessing
import csv
import bs4
import requests
from selenium import webdriver
import os
import time
import random
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from PIL import ImageOps
# OpenAI GPT-4 API Key
OPENAI_API_KEY = " "  # Replace with your API key


# Search parameters

MIN_WIDTH = 1280  # For 720p resolution
MIN_HEIGHT = 720
SHARPNESS_THRESHOLD = 30.0 # Adjust as needed

# Local Tesseract executable path (update if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def is_url_in_csv(image_url,label,urloutput_dir):
    """
    Check if a URL is already present in the CSV file.
    :param output_dir: Directory where the CSV file is saved.
    :param image_url: The URL to check for duplicates.
    :return: True if the URL is in the CSV, False otherwise.
    """
    csv_path = os.path.join(urloutput_dir, f"{label}_approved_images.csv")
    if not os.path.exists(csv_path):
        return False  # If CSV doesn't exist, the URL cannot be a duplicate

    # Read the CSV and check for the URL
    df = pd.read_csv(csv_path)
    return image_url in df["image_url"].values


def save_image_url_to_csv(image_filename, image_url,label,urloutput_dir):
    """
    Save or append the image filename and URL to a CSV file in the output directory.
    :param output_dir: Directory where the CSV file will be saved.
    :param image_filename: Name of the saved image file.
    :param image_url: Original URL of the image.
    """
    csv_path = os.path.join(urloutput_dir, f"{label}_approved_images.csv")

    # Prepare the new row
    new_row = {"image_filename": image_filename, "image_url": image_url}

    # Check if the CSV already exists
    if os.path.exists(csv_path):
        # Append to the existing CSV
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Create a new CSV
        df = pd.DataFrame([new_row])

    # Save the updated DataFrame to CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved to CSV: {csv_path}")

# OpenAI Verifier Class
class OpenAIImageVerifier:
    def __init__(self, api_key):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=api_key)

    def encode_image(self, image_path_or_url):
        """Convert image to base64 string."""
        if image_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url)
            image = Image.open(io.BytesIO(response.content))
        else:
            image = Image.open(image_path_or_url)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def verify_description(self, image_path_or_url, description):
        base64_image = self.encode_image(image_path_or_url)
        prompt = f"""Please verify if this image matches the following description:
        
Description: "{description}"

Please respond with:
1. Whether the image matches(Give yes if its even 30% or more similar to the description / might be similar image) the description (Yes/No)
2. A confidence score (0-100%)"""

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=25
        )
        return response.choices[0].message.content

def is_valid_resolution_and_sharpness(image_path, min_resolution, sharpness_threshold):
    """
    Validate that an image meets the resolution and sharpness requirements.
    :param image_path: Path to the image file.
    :param min_resolution: Tuple (width, height) for minimum resolution (e.g., (1280, 720)).
    :param sharpness_threshold: Minimum sharpness score (higher = sharper).
    :return: True if the image meets the resolution and sharpness requirements, False otherwise.
    """
    try:
        # Check resolution
        with Image.open(image_path) as img:
            width, height = img.size
            min_width, min_height = min_resolution
            if width < min_width or height < min_height:
                print(f"Image resolution too low: {img.size}")
                return False

        # Check sharpness using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        if laplacian_var < sharpness_threshold:
            print(f"Image blurry (sharpness score: {laplacian_var})")
            return False

        return True

    except Exception as e:
        print(f"Error validating image {image_path}: {e}")
        return False

# def resize_image_to_720p(image_path):
#     """
#     Resize the image to 720p (1280x720) resolution while maintaining the aspect ratio.
#     Add padding to fit the image into the target resolution if necessary.
#     :param image_path: Path to the image file.
#     :return: Path to the resized image.
#     """
#     try:
#         with Image.open(image_path) as img:
#             # Resize the image while maintaining aspect ratio
#             img = ImageOps.fit(img, (1280, 720), method=Image.Resampling.LANCZOS)

#             # Save the resized image
#             resized_path = image_path.replace(".jpg", "_resized.jpg")
#             img.save(resized_path, format="JPEG")
#             return resized_path
#     except Exception as e:
#         print(f"Error resizing image {image_path}: {e}")
#         return None

def resize_image_to_720p(image_path):
    """
    Resize the image to 720p (1280x720) resolution while maintaining the aspect ratio.
    Add padding to fit the image into the target resolution if necessary.
    :param image_path: Path to the image file.
    :return: Path to the resized image.
    """
    try:
        with Image.open(image_path) as img:
            # Convert the image to RGB mode to handle potential issues with transparency
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize and pad the image to 1280x720 while maintaining the aspect ratio
            img = ImageOps.fit(img, (1280, 720), method=Image.Resampling.LANCZOS)

            # Save the resized image
            resized_path = image_path.replace(".jpg", "_resized.jpg")
            img.save(resized_path, format="JPEG")
            return resized_path
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return None
    
# Function to download and validate images
def fetch_images(search_term, total_images, output_dir, description, label, urloutput_dir,verifier):

    downloaded_images = 0
    approved_images_csv = os.path.join("Approved_Images", f"{label}_approved_images.csv")
    if os.path.exists(approved_images_csv):
        with open(approved_images_csv, 'r') as file:
            csv_reader = csv.reader(file)
            downloaded_images = sum(1 for row in csv_reader) - 1  # Subtract 1 for the header row


    # Set Chrome options for headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU rendering (for compatibility)
    chrome_options.add_argument("--start-minimized")  # Start the browser minimized
    chrome_options.add_argument("--log-level=3")  # Suppress logs for a cleaner output
    chrome_options.add_argument("--window-size=1920,1080")  # Set a default window size for headless mode
    driver = webdriver.Chrome(options=chrome_options)
    search_URL = f"https://www.google.com/search?q={search_term}&source=lnms&tbm=isch"
    driver.get(search_URL)
    time.sleep(3)
    # Click "Tools"
    tools_xpath = "//*[@id='hdtb-tls']"
    try:
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, tools_xpath))).click()
        time.sleep(4)
    except Exception as e:
        print(f"Error clicking Tools: {str(e)}")

    # Click "Size"
    img_box = driver.find_elements(By.CSS_SELECTOR, 'div.CcNe6e')
    img_box[1].click()
    time.sleep(4)

    # Click "large"
    second_menu_item_xpath = "(//g-menu/g-menu-item)[2]"
    try:
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, second_menu_item_xpath))
        ).click()
        time.sleep(4)
    except Exception as e:
        print(f"Error selecting second menu item: {str(e)}")


    time.sleep(4)
    # Dynamic scrolling to load all containers
    SCROLL_PAUSE_TIME = 2
    try:
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        # Scroll back to the top after reaching the bottom
        driver.execute_script("window.scrollTo(0, 0);")
    except Exception as e:
        print(f"Error during scrolling: {str(e)}")

    time.sleep(4)
    try:
        page_html = driver.page_source
        pageSoup = bs4.BeautifulSoup(page_html, 'html.parser')
        containers = pageSoup.findAll('div', {'class': "eA0Zlc WghbWd FnEtTd mkpRId m3LIae RLdvSe qyKxnc ivg-i PZPZlf GMCzAd"})
        len_containers = len(containers)
        print(f"Total number of containers found: {len_containers}")
    except Exception as e:
        print(f"Error parsing page source: {str(e)}")
        containers = []

    
    while downloaded_images < total_images:
        
        for i in range(1, len_containers + 1):
            if downloaded_images >= total_images:
                break
            if i % 25 == 0:
                continue

            xPath = f"""//*[@id="rso"]/div/div/div[1]/div/div/div[{i}]"""

            try:
                # Wait and click on the image container
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, xPath))).click()
                time.sleep(2)

                # Get the image URL
                retries = 1
                imageURL = None
                for attempt in range(retries):
                    try:
                        imageElement = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located(
                                (By.XPATH, """//*[@id="Sva75c"]/div[2]/div[2]/div/div[2]/c-wiz/div/div[3]/div[1]/a/img[1]""")
                            )
                        )
                        imageURL = imageElement.get_attribute('src')
                        if imageURL and imageURL.startswith("http"):
                            break
                    except Exception as e:
                        if attempt == retries - 1:
                            print(f"Failed to fetch image URL after {retries} attempts for image {i}. Skipping.")
                        else:
                            time.sleep(random.uniform(1, 3))

                if imageURL:
                # Skip if URL is already in the CSV
                    if is_url_in_csv(imageURL,label,urloutput_dir):
                        print(f"Skipping {imageURL} as it is already in the CSV.")
                        continue
                    # Fetch the image
                    img_response = requests.get(imageURL, timeout=10)
                    img_response.raise_for_status()
                    img = Image.open(io.BytesIO(img_response.content))

                    # Save temporarily to check sharpness
                    temp_path = os.path.join(output_dir, "temp.jpg")
                    img.save(temp_path)

                    # Resize the image to 720p if necessary
                    temp_path = resize_image_to_720p(temp_path)


                    # Validate resolution and sharpness
                    if not is_valid_resolution_and_sharpness(temp_path, (MIN_WIDTH, MIN_HEIGHT), SHARPNESS_THRESHOLD):
                        print(f"Skipping {imageURL} due to resolution or sharpness issues.")
                        os.remove(temp_path)  # Remove temporary file
                        continue

                    # Verify description with OpenAI
                    verification_result = verifier.verify_description(imageURL, description)
                    print(f"Verification result for {imageURL}: {verification_result}")
                    if "No" in verification_result:  # Adjust based on GPT-4 response format
                        print(f"Skipping {imageURL} due to description mismatch.--->>>{label}")
                        os.remove(temp_path)  # Remove temporary file
                        continue

                    # Save the image
                    img_filename = os.path.join(output_dir, f"{label}_Image_{downloaded_images + 1:03}.jpg")
                    os.rename(temp_path, img_filename)
                    downloaded_images += 1
                    print(f"Downloaded **********************___*****************************: {img_filename}")
                    
                    # Save the approved image URL to CSV
                    save_image_url_to_csv( os.path.basename(img_filename), imageURL,label,urloutput_dir)

            except Exception as e:
                print(f"Couldn't download an image {i}, continuing downloading the next one. Error: {str(e)}")
                continue


    print(f"Downloaded {downloaded_images} images.")

def process_search_term(row):
    """
    Process a single search term from the Excel row.
    :param row: A row from the Excel file with the required parameters.
    """
    try:
        # Extract parameters from the row
        category_name = row["Category Name"].replace(' ','').replace("->", "_")
        # print(f"!!!!!!!!!!!!!!!!!!Processing: {category_name}")
        total_images = int(row["COUNT"])
        output_dir = os.path.join(category_name)  # Use category name as the folder
        urloutput_dir = os.path.join('Approved_Images')  # Use category name as the folder
        label = row["Label"].replace(" ", "")
        description = row["Definition"]


        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(urloutput_dir, exist_ok=True)
        
        # Set the search term
        search_term = row["Label"]

        # Initialize verifier
        verifier = OpenAIImageVerifier(api_key=OPENAI_API_KEY)

        # Fetch images for this search term
        print(f"Processing: {search_term}")
        fetch_images(
            search_term=search_term,
            total_images=total_images,
            output_dir=output_dir,
            urloutput_dir=urloutput_dir,
            verifier=verifier,
            description=description,
            label=label
        )

    except Exception as e:
        print(f"Error processing row : {e}")

# Main function
if __name__ == "__main__":
    excel_file = "test.xlsx"  # Replace with your Excel file name
    data = pd.read_excel(excel_file)
    categories = ['Adult Video Games']
    pattern = '|'.join(categories)
    data = data[data["Category Name"].astype(str).str.contains(pattern, case=False, na=False)]
    for _, row in data.iterrows():
        process_search_term(row)
    
    #  Use multiprocessing to process each row in parallel
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.map(process_search_term, [row for _, row in data.iterrows()])