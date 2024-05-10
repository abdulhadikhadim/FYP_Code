# FYP_Code

This repository contains the code for a project that involves processing and analyzing video frames and images using a deep learning model.

## Getting Started

To get started with the code, follow the instructions below:

1. Clone the repository to your local machine:
```
git clone https://github.com/abdulhadikhadim/FYP_Code.git
```
2. Navigate to the project directory:
```
cd FYP_Code
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
4. Update the file paths in the code to match your local directory structure. The files that require path updates are:
   - testunseen.py
   - train.py
   - test.py
   - website file (HTML) in the public folder

## Usage

To run the code, use the following commands:

1. To train the deep learning model, navigate to the UNET code's folder and run:
```
python train.py
```
2. To test the model on a new dataset, navigate to the project's root directory and run:
```
python test.py
```
3. To process and analyze video frames, navigate to the project's root directory and run:
```
python testunseen.py
```
4. To view the results, navigate to the results directory specified in the code or open the website (HTML file) in a web browser.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/abdulhadikhadim/FYP_Code/blob/main/LICENSE) file for details.
