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
from PIL import ImageOps
# OpenAI GPT-4 API Key
OPENAI_API_KEY = " "  # Replace with your API key

# Bing Image Search API details
BING_API_KEY = " "  # Replace with your Bing API key
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"

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
1. Whether the image matches(Give yes if its even 5% or more similar to the description / might be similar image) the description (Yes/No)
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
    
def fetch_images(search_term, total_images, output_dir, description, label, urloutput_dir, verifier):
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": search_term, "count": 150, "imageType": "photo", "size": "Large", "safeSearch": "Off"}

    downloaded_images = 0
    offset = 0
    approved_images_csv = os.path.join("Approved_Images", f"{label}_approved_images.csv")
    if os.path.exists(approved_images_csv):
        with open(approved_images_csv, 'r') as file:
            csv_reader = csv.reader(file)
            downloaded_images = sum(1 for row in csv_reader) - 1  # Subtract 1 for the header row

    while downloaded_images < total_images:
        params["offset"] = offset
        response = requests.get(BING_ENDPOINT, headers=headers, params=params)
        response.raise_for_status()
        results = response.json()

        for img_data in results.get("value", []):
            if downloaded_images >= total_images:
                break
            img_url = img_data["contentUrl"]
            try:

                # Skip if URL is already in the CSV
                if is_url_in_csv(img_url, label, urloutput_dir):
                    print(f"Skipping {img_url} as it is already in the CSV.")
                    continue

                # Fetch the image
                img_response = requests.get(img_url, timeout=10)
                img_response.raise_for_status()
                img = Image.open(io.BytesIO(img_response.content))

                # Save temporarily to check sharpness and resolution
                temp_path = os.path.join(output_dir, "temp.jpg")
                img.save(temp_path)

                # Resize the image to 720p if necessary
                temp_path = resize_image_to_720p(temp_path)

                # Validate resolution and sharpness
                if not is_valid_resolution_and_sharpness(temp_path, (MIN_WIDTH, MIN_HEIGHT), SHARPNESS_THRESHOLD):
                    print(f"Skipping {img_url} due to resolution or sharpness issues.")
                    os.remove(temp_path)  # Remove temporary file
                    continue

                # Verify description with OpenAI
                verification_result = verifier.verify_description(temp_path, description)
                print(f"Verification result for {img_url}: {verification_result}")
                if "No" in verification_result:  # Adjust based on GPT-4 response format
                    print(f"Skipping {img_url} due to description mismatch.")
                    os.remove(temp_path)  # Remove temporary file
                    continue

                # Save the image
                img_filename = os.path.join(output_dir, f"{label}_Image_{downloaded_images + 1:03}.jpg")
                os.rename(temp_path, img_filename)
                downloaded_images += 1
                print(f"Downloaded: {img_filename}")
                
                # Save the approved image URL to CSV
                save_image_url_to_csv(os.path.basename(img_filename), img_url, label, urloutput_dir)

            except Exception as e:
                print(f"Error downloading {img_url}: {e}")

        offset += len(results.get("value", []))

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
    # Use multiprocessing to process each row in parallel
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.map(process_search_term, [row for _, row in data.iterrows()])
        

