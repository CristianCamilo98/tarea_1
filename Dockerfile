# Use the official Python base image
FROM python:3.14.0-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
  apt-get install -y build-essential && \
  pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

# Copy the entire project into the container
COPY . ./

# Command to start the application, by default assuming user may start examples
CMD ["python3", "examples/interactive_portfolio_analysis.py"]

