"""In-memory image decoding for HTTP requests."""

import base64
import binascii
import io

from PIL import Image, UnidentifiedImageError

from .api import validation_error


def decode_image(value):
    """Decode a base64 string or image data URL into an RGB Pillow image."""
    try:
        if value.startswith("data:"):
            header, value = value.split(",", 1)
            if not header.startswith("data:image/") or not header.endswith(";base64"):
                raise validation_error("image", "Expected an image data URL with base64 encoding")
        raw = base64.b64decode(value)
        with Image.open(io.BytesIO(raw)) as image:
            if Image.MAX_IMAGE_PIXELS is not None and image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise validation_error("image", "The image exceeds the decoder's pixel limit")
            return image.convert("RGB")
    except (ValueError, binascii.Error) as error:
        raise validation_error("image", "Expected base64 image data or a base64 image data URL") from error
    except (UnidentifiedImageError, OSError) as error:
        raise validation_error("image", "The image could not be decoded") from error
    except Image.DecompressionBombError as error:
        raise validation_error("image", "The image exceeds the decoder's pixel limit") from error
