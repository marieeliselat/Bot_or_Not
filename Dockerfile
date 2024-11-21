FROM python:3

# Install required Python packages
RUN pip install requests
RUN pip install pydantic
RUN pip install textblob
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install matplotlib
RUN pip install pandas
RUN pip install openai==0.28

# Copy the application files into the container
COPY . .

# Ensure the entrypoint script has execute permissions
RUN chmod +x run.sh

# Run the application with the shell script
CMD ["sh", "run.sh"]
