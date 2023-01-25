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
#COPY pyproject.toml .
#COPY requirements.txt .
COPY requirements_dev.txt .

RUN pip install -r requirements_dev.txt

# Install application into container
COPY . .

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /minerva
USER appuser

CMD [ "python", "--version" ]
