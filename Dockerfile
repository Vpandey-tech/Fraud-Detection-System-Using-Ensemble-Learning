FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the entire project to the container
COPY . /app

# Install dependencies from requirements.txt
# We upgrade pip first to avoid errors
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create a non-root user (Hugging Face security best practice)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Expose the port Hugging Face Spaces expects (7860)
EXPOSE 7860

# Give execution rights and run the start script
CMD ["bash", "start_hf.sh"]
