import base64
import io

from PIL import Image
import pytest

from fuzev.api import APIError, api_error
from fuzev.images import decode_image
from fuzev.service_contracts import GenerationRequest, ServiceDecisionRequest
from fuzev.systemone import SystemOneRequest


def encoded_image(format="PNG"):
    with Image.new("L", (4, 3), 80) as image, io.BytesIO() as buffer:
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.parametrize("format,data_url", [("PNG", True), ("JPEG", True), ("PNG", False)])
def test_standard_image_inputs_decode_to_rgb(format, data_url):
    value = encoded_image(format)
    if data_url:
        value = f"data:image/{format.lower()};base64,{value}"
    with decode_image(value) as image:
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (4, 3)
        assert image.getpixel((0, 0)) == (80, 80, 80)


def test_image_request_fields_accept_strings():
    value = "data:image/png;base64," + encoded_image()
    requests = [
        SystemOneRequest(state="A photo", questions={"q": {"type": "noul"}}, image=value),
        ServiceDecisionRequest(
            model="actual-model", primitive="choice", messages=[{"role": "user", "content": "A photo"}],
            candidates=[{"id": "a", "text": "A"}, {"id": "b", "text": "B"}], image=value,
        ),
        GenerationRequest(model="actual-model", prompt="Describe the photo", image=value),
    ]
    assert all(request.model_dump()["image"] == value for request in requests)


@pytest.mark.parametrize("value,message", [
    ("not base64!", "Expected base64 image data or a base64 image data URL"),
    (base64.b64encode(b"not an image").decode("ascii"), "The image could not be decoded"),
    ("data:text/plain;base64,aGVsbG8=", "Expected an image data URL with base64 encoding"),
])
def test_bad_image_inputs_return_field_errors(value, message):
    with pytest.raises(APIError) as error:
        decode_image(value)
    status, body = api_error(error.value)
    assert status == 422
    assert body["detail"][0]["loc"] == ["body", "image"]
    assert body["detail"][0]["msg"] == message


def test_multiframe_image_uses_first_frame_and_accepts_base64_whitespace():
    with Image.new("RGB", (4, 3), "red") as first, Image.new("RGB", (4, 3), "blue") as second:
        with io.BytesIO() as buffer:
            first.save(buffer, format="GIF", save_all=True, append_images=[second], duration=100)
            value = base64.b64encode(buffer.getvalue()).decode("ascii")
    with decode_image("\n".join(value[i:i + 20] for i in range(0, len(value), 20))) as image:
        assert image.size == (4, 3)
        assert image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("pixel_limit", [8, 4])
def test_pillow_decompression_bombs_are_rejected(monkeypatch, pixel_limit):
    value = encoded_image()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pixel_limit)
    with pytest.raises(APIError, match="pixel limit"):
        decode_image(value)
