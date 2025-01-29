import os
import cv2
from numpy.random import random, randint
from pdf2image import convert_from_path
from augraphy import *
from PIL import Image
import concurrent.futures

# Define paths
base_pdf_path = "data/base_document.pdf"
output_dir = "data/documents_to_check/augmented_documents"


def make_pipeline() -> AugraphyPipeline:
    return default_augraphy_pipeline()


def simulate_scanned_document_pipeline() -> AugraphyPipeline:
    """
    Returns a pipeline configured to simulate a typical scanned printed document
    with mild artifacts like noise, drum lines, brightness changes, and compression.
    """
    return AugraphyPipeline([
        # 2. Add subtle noise to emulate sensor/scan noise
        # SubtleNoise(p=0.5),

        # 3. Mildly change brightness/contrast in a texturized way
        # BrightnessTexturize(texturize_range=(0.8, 0.9),deviation=0.05, p=0.5),

        # 4. Slight blur to mimic a non-perfect scan focus
        BadPhotoCopy(p=0.5),

        # 5. Compress the image to replicate typical scan compression
        #    or scanning software's JPEG output
        # Jpeg(quality_range=(30, 60), p=0.5),
        # LowInkPeriodicLines(p=0.5),
    ])


# Function to augment a single page image
def augment_image(image_path, pipeline):
    """
    Applies the Augraphy pipeline to an image.
    """
    # Load the image with OpenCV
    image = cv2.imread(image_path)
    augmented_image = pipeline(image)  # Apply the pipeline
    return augmented_image


# Function to create 200 augmented copies
def create_augmented_documents(base_pdf, output_dir, thread_id, num_copies=200):
    """
    Generates augmented copies of a base PDF document.
    """
    # Define the Augraphy pipeline
    pipeline = simulate_scanned_document_pipeline()
    # Convert the PDF into images (one per page)
    images = convert_from_path(base_pdf)

    # Process each copy
    for i in range(1, num_copies + 1):
        try:
            augmented_images = []

            for page_num, page_image in enumerate(images):
                # Save the page image temporarily as a PNG for OpenCV compatibility
                page_id = randint(10000)
                temp_image_path = f"temp_page_{page_id}.png"
                page_image.save(temp_image_path)

                # Apply Augraphy to the page
                augmented_image = augment_image(temp_image_path, pipeline)

                # Convert augmented OpenCV image back to PIL for saving as PDF
                augmented_pil_image = Image.fromarray(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
                augmented_images.append(augmented_pil_image)

                # Clean up temporary file
                os.remove(temp_image_path)

            # Save all augmented pages into a single PDF
            pdf_num = i * (thread_id+1)
            output_pdf_path = os.path.join(output_dir, f"augmented_document_{pdf_num:03d}.pdf")
            augmented_images[0].save(output_pdf_path, save_all=True, append_images=augmented_images[1:])

            print(f"Augmented document {i}/{num_copies} saved to: {output_pdf_path}")

        except Exception as e:
            print(f"Error processing document {i}: {e}")


# Number of augmented copies per thread (split evenly)
NUM_COPIES = 100
NUM_THREADS = 4
copies_per_thread = NUM_COPIES // NUM_THREADS


# Define function wrapper
def worker(thread_id):
    """Each thread will generate a portion of the total augmented copies."""
    print(f"Thread {thread_id} started")
    create_augmented_documents(base_pdf_path, output_dir, thread_id, num_copies=copies_per_thread)
    print(f"Thread {thread_id} completed")


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Use ProcessPoolExecutor for multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(worker, i) for i in range(NUM_THREADS)]
        concurrent.futures.wait(futures)  # Wait for all threads to finish

    print("All threads completed processing.")
