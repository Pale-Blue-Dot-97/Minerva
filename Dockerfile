FROM python:3.11.3 as base

# Update package listings, install git and OpenCV.
RUN apt-get -y update \
    && apt-get install git -y --no-install-recommends \
    && apt-get install -y --no-install-recommends python3-opencv -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensures easier to read stdout.
ENV PYTHONUNBUFFERED=1

FROM base AS python-deps

# Copy needed files.
COPY setup.py .
COPY setup.cfg .
COPY requirements/requirements_dev.txt .

RUN pip install -r requirements_dev.txt

# Install application into container
COPY . .

CMD [ "/bin/bash/" ]
