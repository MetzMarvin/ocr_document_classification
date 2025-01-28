import os
import cv2
from pdf2image import convert_from_path
from augraphy import *
from PIL import Image

# Define paths
base_pdf_path = "data/documents_to_check/base_document.pdf"
output_dir = "data/documents_to_check/augmented_documents"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Define the Augraphy pipeline
pipeline = default_augraphy_pipeline()


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
def create_augmented_documents(base_pdf, output_dir, num_copies=200):
    """
    Generates augmented copies of a base PDF document.
    """
    # Convert the PDF into images (one per page)
    images = convert_from_path(base_pdf)

    # Process each copy
    for i in range(1, num_copies + 1):
        augmented_images = []

        for page_num, page_image in enumerate(images):
            # Save the page image temporarily as a PNG for OpenCV compatibility
            temp_image_path = f"temp_page_{page_num}.png"
            page_image.save(temp_image_path)

            # Apply Augraphy to the page
            augmented_image = augment_image(temp_image_path, pipeline)

            # Convert augmented OpenCV image back to PIL for saving as PDF
            augmented_pil_image = Image.fromarray(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
            augmented_images.append(augmented_pil_image)

            # Clean up temporary file
            os.remove(temp_image_path)

        # Save all augmented pages into a single PDF
        output_pdf_path = os.path.join(output_dir, f"augmented_document_{i:03d}.pdf")
        augmented_images[0].save(output_pdf_path, save_all=True, append_images=augmented_images[1:])

        print(f"Augmented document {i}/{num_copies} saved to: {output_pdf_path}")


# Generate 200 augmented documents
create_augmented_documents(base_pdf_path, output_dir, num_copies=200)
