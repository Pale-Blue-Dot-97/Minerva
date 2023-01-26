FROM python:3.10.9 as base

# Update package listings, install git and OpenCV.
RUN apt-get -y update && apt-get install git && apt-get install -y python3-opencv -y

# Ensures easier to read stdout.
ENV PYTHONUNBUFFERED=1

FROM base AS python-deps

# Copy needed files.
COPY Pipfile .
COPY Pipfile.lock .
COPY setup.py .
COPY setup.cfg .
COPY requirements_dev.txt .

RUN pip install -r requirements_dev.txt

# Install application into container
COPY . .

ARG USER_ID
ARG GROUP_ID

# Creates a user and group for use in container from the user and group used to construct the image.
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user

CMD [ "/bin/bash/" ]
