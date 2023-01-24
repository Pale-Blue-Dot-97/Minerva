FROM ubuntu:22.04 as base

# Update package listings.
RUN apt-get -y update

# Install Python 3.10.
RUN apt-get install python3.10 -y

# Install pip.
RUN apt-get install python3-pip -y

# Install git.
RUN apt-get install git -y

# Setup env.
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1

FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN pip install pipenv
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Install python dependencies in /.venv
COPY Pipfile .
COPY Pipfile.lock .
COPY setup.py .
COPY setup.cfg .
COPY pyproject.toml .
COPY requirements.txt .
COPY requirements_dev.txt .

# pipenv install the dev version from Pipfile.lock,
# exports lock requirements to file,
# and builds a wheel of `minerva`.
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --dev
# RUN pipenv lock -r > requirements.txt
# RUN pipenv run python setup.py bdist_wheel

FROM base AS runtime

# COPY --from=python-deps dist/*.whl .

# RUN pip install *.whl
# RUN rm -f *.whl

# Copy virtual env from python-deps stage
COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

# Install application into container
COPY . .
