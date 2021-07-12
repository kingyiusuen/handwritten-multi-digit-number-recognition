from pydantic import BaseModel, validator


class Image(BaseModel):
    """Schema for user inputs from the frontend."""

    image: str

    @validator("image")
    def image_must_not_be_empty(cls, value):
        """Validate the input."""
        if not len(value):
            raise ValueError("image cannot be empty.")
        return value
