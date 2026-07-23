from pdf2image import convert_from_bytes
import io

POPPLER_PATH = r"C:\poppler\Library\bin"


def convert_pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    """
    Convert PDF bytes into a list of image bytes.
    Input:
        pdf_bytes -> raw PDF file bytes
    Output:
        list of images in bytes format
    """

    # Convert PDF pages into PIL images
    pages = convert_from_bytes(
        pdf_bytes,
        dpi=300,
        poppler_path=POPPLER_PATH
    )

    image_bytes_list = []

    for page in pages:

        # Create temporary memory buffer
        buffer = io.BytesIO()

        # Save page as PNG into memory
        page.save(
            buffer,
            format="PNG"
        )

        # Move buffer pointer to beginning
        buffer.seek(0)

        # Store image bytes
        image_bytes_list.append(
            buffer.read()
        )

    return image_bytes_list