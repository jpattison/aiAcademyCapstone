This project was made to complete the capstone project for the AI Academy project.
By: Jeremy Pattison jeremy.pattison@ibm.com

This codebase has been designed to run as a self contained dockerfile.

EDA and model analysis reports are found in the two associated jupyter notebooks.

To run:

Build and run the docker image to open the jupyter notebooks. Note the jupyter is being run out of a shared volume to save and see changes on your local machine.

CMD: docker build -t capstone_jeremy:1 .
CMD: docker run -it -v "$(pwd)":/home/jovyan/ --rm -p 8888:8888 --name capstone capstone_jeremy:1

To run the flask application you will need to perform a docker exec action.

CMD: docker exec -it capstone python app.py 

