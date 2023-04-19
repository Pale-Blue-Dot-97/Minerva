FROM python:3.11.3 as base

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
COPY requirements/requirements_dev.txt .

RUN pip install -r requirements_dev.txt

# Install application into container
COPY . .

CMD [ "/bin/bash/" ]
